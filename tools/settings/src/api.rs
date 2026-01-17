//! API calls for dynamic option fetching

use anyhow::Result;
use serde::Deserialize;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModel>,
}

#[derive(Debug, Deserialize)]
struct OllamaModel {
    name: String,
}

/// Fetch available Ollama models from the API
pub fn fetch_ollama_models(base_url: &str) -> Result<Vec<String>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    let url = format!("{}/api/tags", base_url);

    let response: OllamaTagsResponse = client.get(&url).send()?.json()?;

    Ok(response.models.into_iter().map(|m| m.name).collect())
}

/// Fetch available TTS models from `tts --list_models`
pub fn fetch_tts_models() -> Result<Vec<String>> {
    let output = Command::new("tts").arg("--list_models").output()?;

    if !output.status.success() {
        anyhow::bail!("tts --list_models failed");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut models = Vec::new();

    // Parse the output - models are listed as paths like "tts_models/en/vctk/vits"
    for line in stdout.lines() {
        let line = line.trim();
        // Skip headers and empty lines
        if line.is_empty() || line.starts_with("Name") || line.contains(':') && !line.contains('/')
        {
            continue;
        }
        // Lines starting with space and containing model paths
        if line.contains("tts_models/") || line.contains("vocoder_models/") {
            // Extract just the model path
            let model = line.trim_start_matches(" >").trim();
            if !model.is_empty() {
                models.push(model.to_string());
            }
        }
    }

    // If parsing didn't find any, return some common defaults
    if models.is_empty() {
        models = vec![
            "tts_models/en/vctk/vits".to_string(),
            "tts_models/en/ljspeech/tacotron2-DDC".to_string(),
            "tts_models/en/ljspeech/glow-tts".to_string(),
            "tts_models/en/ljspeech/tacotron2-DCA".to_string(),
            "tts_models/multilingual/multi-dataset/xtts_v2".to_string(),
        ];
    }

    Ok(models)
}

/// Fetch available TTS speakers for multi-speaker models (VCTK)
pub fn fetch_tts_speakers() -> Result<Vec<String>> {
    // VCTK dataset has speakers p225-p376 (with some gaps)
    // These are the commonly available ones
    let speakers: Vec<String> = vec![
        "p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236",
        "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248",
        "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259",
        "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270",
        "p271", "p272", "p273", "p274", "p275", "p276", "p277", "p278", "p279", "p280", "p281",
        "p282", "p283", "p284", "p285", "p286", "p287", "p288", "p292", "p293", "p294", "p295",
        "p297", "p298", "p299", "p300", "p301", "p302", "p303", "p304", "p305", "p306", "p307",
        "p308", "p310", "p311", "p312", "p313", "p314", "p316", "p317", "p318", "p323", "p326",
        "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345",
        "p347", "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376",
    ]
    .into_iter()
    .map(String::from)
    .collect();

    Ok(speakers)
}
