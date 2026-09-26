//! Plain JSON-shaped values used to pass checkpoint declarations between
//! format readers and model metadata parsers.

/// A decoded configuration value without any JSON or I/O dependency.
#[derive(Debug, Clone, PartialEq)]
pub enum DeclaredValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    List(Vec<DeclaredValue>),
    Null,
}
