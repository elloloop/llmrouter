# Changelog

## [0.3.0](https://github.com/elloloop/llmrouter/compare/v0.2.0...v0.3.0) (2026-05-15)


### Features

* add Speaker, Transcriber, Embedder interfaces + root types ([a8482c3](https://github.com/elloloop/llmrouter/commit/a8482c3b0aeeaf44745df39588b4e878a27b271c))
* **bedrock:** embeddings via InvokeModel for Titan + Cohere ([de3ad80](https://github.com/elloloop/llmrouter/commit/de3ad80dee568f2db5cb132c8598a291ca38a05a))
* **cartesia:** TTS (Speaker) for real-time voice agents ([10d7301](https://github.com/elloloop/llmrouter/commit/10d73010d5412b843d9baf3c96395af15100b560))
* **cohere,mistral,together:** embeddings ([e28ba12](https://github.com/elloloop/llmrouter/commit/e28ba121c5c675ecc4a20e6263acd22370cf80d3))
* **deepgram:** STT via Nova-3 (Transcriber) ([db17bed](https://github.com/elloloop/llmrouter/commit/db17bedd790329a3d23a4f9d20c0828aa3cc1185))
* **elevenlabs:** TTS (Speaker) + STT (Transcriber) ([91ebbf7](https://github.com/elloloop/llmrouter/commit/91ebbf7669f44aef3a09f798813f5097c2e26e32))
* **gemini,vertex:** audio (TTS+STT) + embeddings ([1228a66](https://github.com/elloloop/llmrouter/commit/1228a660fd283b2fee4de14a5eb313afea1b3e47))
* **groq:** Whisper STT via OpenAI-compatible audio endpoint ([0881281](https://github.com/elloloop/llmrouter/commit/08812817c142d1a7f9868a5e4eb3cfbf4c1620d3))
* **openai,azureopenai:** audio (TTS+STT) + embeddings ([2af970e](https://github.com/elloloop/llmrouter/commit/2af970e8e8188decec5ff4f16ea22db82a3323ad))
* **voyage:** Voyage AI embeddings (Embedder) ([a4a3f6a](https://github.com/elloloop/llmrouter/commit/a4a3f6a0ddeded4fad13dac0f6a14390531a0573))

## [0.2.0](https://github.com/elloloop/llmrouter/compare/v0.1.1...v0.2.0) (2026-05-15)


### Features

* **azureopenai:** Azure OpenAI Service provider ([91bd56d](https://github.com/elloloop/llmrouter/commit/91bd56d38942de098da332d98a4bf8bf1f178609))
* **bedrock:** AWS Bedrock provider via Converse Stream API ([ce26369](https://github.com/elloloop/llmrouter/commit/ce2636937336e44f8c89c255c233def87f88f538))
* **cohere:** Cohere v2 Chat provider ([4fbe4b0](https://github.com/elloloop/llmrouter/commit/4fbe4b0968e1f3fcbaca4919667efe81e6032811))
* **gemini:** Google AI Studio Gemini provider (HTTP, no SDK) ([f6a8a02](https://github.com/elloloop/llmrouter/commit/f6a8a022472850b302be48e2eb73ec168818295a))
* **mistral:** Mistral AI chat provider ([8db32aa](https://github.com/elloloop/llmrouter/commit/8db32aafd1c61422495a3013c5f8c4e120d87dc4))
* **openrouter:** OpenRouter provider with attribution headers ([1fbca4b](https://github.com/elloloop/llmrouter/commit/1fbca4bf50895f1a8d660fbc477f61d9f97efcb7))
* **providers:** thin OpenAI-compatible helpers for 7 vendors ([3917dec](https://github.com/elloloop/llmrouter/commit/3917dec46b070e107a5b2a103725fd9d19ff3b63))
* tool use, thinking output, prompt caching, multimodal helpers ([488f4b1](https://github.com/elloloop/llmrouter/commit/488f4b1e0d5c0cca15bc1f4c083af5005f44cde1))
* **vertex:** Google Vertex AI Gemini provider ([18d0711](https://github.com/elloloop/llmrouter/commit/18d07114b878c5d29d2cf33375f11fed9a568945))


### Bug Fixes

* **vertex:** skip TestNew_Succeeds when ADC unavailable ([5e3068e](https://github.com/elloloop/llmrouter/commit/5e3068e4404856660a2d29e5cf9bfe7c5757fd8a))
