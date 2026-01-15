//! Kafka producer for sending prompts to ai_in topic

use anyhow::{Context, Result};
use chrono::Local;
use rdkafka::config::ClientConfig;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde::Serialize;
use std::time::Duration;

/// Prompt message format for ai_in topic
#[derive(Debug, Serialize)]
pub struct PromptMessage {
    pub text: String,
    pub id: String,
}

impl PromptMessage {
    pub fn new(text: String) -> Self {
        let id = Local::now().format("%Y%m%d_%H%M%S_%f").to_string();
        Self { text, id }
    }
}

/// Kafka producer for sending to ai_in topic
pub struct KafkaProducer {
    producer: FutureProducer,
    topic: String,
}

impl KafkaProducer {
    pub fn new(bootstrap_servers: &str, topic: &str) -> Result<Self> {
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", bootstrap_servers)
            .set("message.timeout.ms", "10000")
            .set("queue.buffering.max.ms", "0") // Send immediately
            .create()
            .context("Failed to create Kafka producer")?;

        Ok(Self {
            producer,
            topic: topic.to_string(),
        })
    }

    /// Send a prompt message to the ai_in topic
    pub async fn send_prompt(&self, text: &str) -> Result<()> {
        let message = PromptMessage::new(text.to_string());
        let payload = serde_json::to_string(&message)?;

        let record = FutureRecord::to(&self.topic)
            .payload(&payload)
            .key(&message.id);

        self.producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(e, _)| anyhow::anyhow!("Failed to send message: {}", e))?;

        Ok(())
    }
}

/// Create a producer with default settings from config
#[allow(dead_code)]
pub fn create_producer(bootstrap_servers: &str) -> Result<KafkaProducer> {
    KafkaProducer::new(bootstrap_servers, "ai_in")
}
