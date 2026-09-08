use std::{ /* checkpoint files */ fs };

pub fn read_weights(p: &str) -> std::io::Result<Vec<u8>> {
    fs::read(p)
}
