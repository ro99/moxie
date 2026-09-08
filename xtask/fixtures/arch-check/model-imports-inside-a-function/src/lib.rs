pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> {
    use std::{fs};
    fs::read(p)
}
