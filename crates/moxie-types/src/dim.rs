//! Checked symbolic dimensions.
//!
//! Document 02: "Unknown dimensions use checked symbolic expressions, not magic
//! maximum-context constants." A graph is built before the context length is
//! known, so a shape may reference a symbol (`seq`, `ctx`, `rows`) that is bound
//! at plan time. Every arithmetic step is checked; overflow is an error, never a
//! wrap, because a wrapped dimension becomes a buffer size.

use core::fmt;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymbolId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DimError {
    /// Checked arithmetic overflowed. A dimension that wraps becomes a buffer
    /// size that is too small; this is never tolerated.
    Overflow,
    /// A symbol was used without a binding.
    Unbound(SymbolId),
    /// Division by zero, or a non-exact division where exactness is required.
    NotDivisible { value: u64, by: u64 },
}

impl fmt::Display for DimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimError::Overflow => write!(f, "dimension arithmetic overflowed"),
            DimError::Unbound(s) => write!(f, "symbol {} is unbound", s.0),
            DimError::NotDivisible { value, by } => {
                write!(f, "{value} is not exactly divisible by {by}")
            }
        }
    }
}

impl std::error::Error for DimError {}

/// A dimension expression. Deliberately closed: no arbitrary callbacks, so a
/// plan can reason about shapes without executing model code (document 02).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Const(u64),
    Symbol(SymbolId),
    Add(Box<Dim>, Box<Dim>),
    Mul(Box<Dim>, Box<Dim>),
    /// Exact division. Errors rather than truncating, because a silently
    /// truncated dimension is a partition bug (document 04, TP padding rules).
    DivExact(Box<Dim>, u64),
    /// Ceiling division, for page and chunk counts where padding is intended.
    DivCeil(Box<Dim>, u64),
}

impl Dim {
    pub fn constant(v: u64) -> Self {
        Dim::Const(v)
    }

    pub fn symbol(id: SymbolId) -> Self {
        Dim::Symbol(id)
    }

    pub fn div_exact(self, by: u64) -> Self {
        Dim::DivExact(Box::new(self), by)
    }

    pub fn div_ceil(self, by: u64) -> Self {
        Dim::DivCeil(Box::new(self), by)
    }

    /// Evaluate against a binding. Every step is checked.
    pub fn eval(&self, bindings: &SymbolTable) -> Result<u64, DimError> {
        match self {
            Dim::Const(v) => Ok(*v),
            Dim::Symbol(s) => bindings.get(*s).ok_or(DimError::Unbound(*s)),
            Dim::Add(a, b) => a
                .eval(bindings)?
                .checked_add(b.eval(bindings)?)
                .ok_or(DimError::Overflow),
            Dim::Mul(a, b) => a
                .eval(bindings)?
                .checked_mul(b.eval(bindings)?)
                .ok_or(DimError::Overflow),
            Dim::DivExact(a, by) => {
                let v = a.eval(bindings)?;
                if *by == 0 {
                    return Err(DimError::NotDivisible { value: v, by: *by });
                }
                if v % by != 0 {
                    return Err(DimError::NotDivisible { value: v, by: *by });
                }
                Ok(v / by)
            }
            Dim::DivCeil(a, by) => {
                let v = a.eval(bindings)?;
                if *by == 0 {
                    return Err(DimError::NotDivisible { value: v, by: *by });
                }
                Ok(v.div_ceil(*by))
            }
        }
    }

    /// Symbols this expression depends on, so a plan cache key can include them.
    pub fn symbols(&self, out: &mut Vec<SymbolId>) {
        match self {
            Dim::Const(_) => {}
            Dim::Symbol(s) => {
                if !out.contains(s) {
                    out.push(*s);
                }
            }
            Dim::Add(a, b) | Dim::Mul(a, b) => {
                a.symbols(out);
                b.symbols(out);
            }
            Dim::DivExact(a, _) | Dim::DivCeil(a, _) => a.symbols(out),
        }
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Dim::Const(_))
    }
}

// `Add`/`Mul` rather than inherent `add`/`mul` methods: a dimension expression
// reads naturally as arithmetic, and the operators cannot be confused with the
// standard traits the way similarly named inherent methods can.
impl core::ops::Add for Dim {
    type Output = Dim;
    fn add(self, rhs: Dim) -> Dim {
        Dim::Add(Box::new(self), Box::new(rhs))
    }
}

impl core::ops::Mul for Dim {
    type Output = Dim;
    fn mul(self, rhs: Dim) -> Dim {
        Dim::Mul(Box::new(self), Box::new(rhs))
    }
}

#[derive(Debug, Clone, Default)]
pub struct SymbolTable {
    bindings: HashMap<SymbolId, u64>,
    names: HashMap<SymbolId, String>,
}

impl SymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn declare(&mut self, id: SymbolId, name: impl Into<String>) {
        self.names.insert(id, name.into());
    }

    pub fn bind(&mut self, id: SymbolId, value: u64) {
        self.bindings.insert(id, value);
    }

    pub fn get(&self, id: SymbolId) -> Option<u64> {
        self.bindings.get(&id).copied()
    }

    pub fn name(&self, id: SymbolId) -> Option<&str> {
        self.names.get(&id).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: SymbolId = SymbolId(0);

    fn table(seq: u64) -> SymbolTable {
        let mut t = SymbolTable::new();
        t.declare(SEQ, "seq");
        t.bind(SEQ, seq);
        t
    }

    #[test]
    fn evaluates_a_kv_byte_expression() {
        // Document 03's conservative bound, as a shape rather than a constant:
        // layers * tokens * kv_heads * (key_dim + value_dim) * 2
        let expr = Dim::constant(78)
            * Dim::symbol(SEQ)
            * Dim::constant(8)
            * Dim::constant(192 + 192)
            * Dim::constant(2);
        assert_eq!(
            expr.eval(&table(32_768)).unwrap(),
            78 * 32_768 * 8 * 384 * 2
        );
    }

    #[test]
    fn overflow_is_an_error_not_a_wrap() {
        let expr = Dim::constant(u64::MAX) * Dim::constant(2);
        assert_eq!(expr.eval(&table(1)), Err(DimError::Overflow));

        let expr = Dim::constant(u64::MAX) + Dim::constant(1);
        assert_eq!(expr.eval(&table(1)), Err(DimError::Overflow));
    }

    #[test]
    fn unbound_symbol_is_an_error_not_a_default() {
        // A missing context length must not silently become zero or a maximum.
        let expr = Dim::symbol(SymbolId(7));
        assert_eq!(expr.eval(&table(1)), Err(DimError::Unbound(SymbolId(7))));
    }

    #[test]
    fn non_divisible_partition_is_rejected() {
        // Document 04: non-divisible dimensions use checked padding or an
        // explicit unsupported combination -- never a truncating divide.
        let heads = Dim::constant(64);
        assert_eq!(heads.clone().div_exact(8).eval(&table(1)).unwrap(), 8);
        assert_eq!(
            heads.div_exact(5).eval(&table(1)),
            Err(DimError::NotDivisible { value: 64, by: 5 })
        );
    }

    #[test]
    fn div_ceil_pads_pages() {
        let toks = Dim::symbol(SEQ).div_ceil(256);
        assert_eq!(toks.eval(&table(32_768)).unwrap(), 128);
        assert_eq!(toks.eval(&table(32_769)).unwrap(), 129);
        assert_eq!(toks.eval(&table(0)).unwrap(), 0);
    }

    #[test]
    fn division_by_zero_is_an_error() {
        assert!(Dim::constant(4).div_exact(0).eval(&table(1)).is_err());
        assert!(Dim::constant(4).div_ceil(0).eval(&table(1)).is_err());
    }

    #[test]
    fn symbols_are_collected_for_cache_keys() {
        let expr = Dim::symbol(SEQ) * Dim::symbol(SymbolId(3)) + Dim::symbol(SEQ);
        let mut out = Vec::new();
        expr.symbols(&mut out);
        out.sort_unstable();
        assert_eq!(out, vec![SymbolId(0), SymbolId(3)]);
    }
}
