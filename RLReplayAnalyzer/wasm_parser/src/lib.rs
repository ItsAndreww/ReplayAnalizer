use wasm_bindgen::prelude::*;
use boxcars::ParserBuilder;

#[wasm_bindgen]
pub fn parse_replay(data: &[u8]) -> Result<String, JsValue> {
    // 1. Згодовуємо масив байтів парсеру boxcars
    let replay = ParserBuilder::new(data)
        .on_error_check_crc()
        .parse()
        .map_err(|e| JsValue::from_str(&format!("Помилка парсингу: {:?}", e)))?;

    // 2. Перетворюємо структуру в рядок JSON
    serde_json::to_string(&replay)
        .map_err(|e| JsValue::from_str(&format!("Помилка генерації JSON: {:?}", e)))
}