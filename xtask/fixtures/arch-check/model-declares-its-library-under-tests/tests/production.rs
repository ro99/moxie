pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(p)
}
