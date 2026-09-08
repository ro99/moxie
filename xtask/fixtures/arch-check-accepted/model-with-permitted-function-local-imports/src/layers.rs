use moxie_graph::Op;

pub fn ops() -> Vec<Op> {
    vec![Op::Embedding, Op::Linear, Op::RmsNorm]
}

pub fn count() -> usize {
    let ops = self::ops();
    ops.len()
}
