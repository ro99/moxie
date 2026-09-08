pub fn read(path: &str) -> std::io::Result<Vec<u8>> {
    std::fs::read(path)
}
