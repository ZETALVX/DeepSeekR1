# -*- coding: utf-8 -*-

import os
from llama_cpp import Llama

# 📌 Set the model path
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-7B-Q5_K_M.gguf"
SAVE_PATH = r"E:\ia" 
MODEL_PATH = os.path.join(SAVE_PATH, MODEL_NAME)
temperatureModel=0.3

# 📌 File path to save conversation history
CHAT_HISTORY_FILE = "chat_history.txt"

# 📌 Verify that the model exists before uploading it
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: The model was not found in {MODEL_PATH}.")
    exit(1)

# 📌 Loading GGUF model with llama.cpp
print(f"🚀 Loading the model {MODEL_NAME}...")
llm = Llama(model_path=MODEL_PATH, n_ctx=32768, n_batch=128, verbose=False) #n_ctx=32768, n_batch=128 Change the values ​​according to your hardware.
print("✅ Model uploaded successfully!")

# 📌 Function to save the conversation to a file
def save_to_history(user_input, bot_reply):
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"User: {user_input}\nBot: {bot_reply}\n\n")

# 📌 Let's read the previous history, if it exists
if os.path.exists(CHAT_HISTORY_FILE):
    print("\n📜 Previous conversation history:\n")
    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        print(f.read())

# 📌 Chat Loop
print("\n🤖 Chatbot started! Write a message to get started (type 'exit' to exit).")

while True:
    user_input = input("\nYou: ")
    
    if user_input.lower() in ["exit", "quit", "esci"]:
        print("👋 Chat ended. Goodbye!")
        break

    # 📌 We generate the response with the improved prompt to avoid reflections and insecurities
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a virtual assistant named 'Zetalbot'. Respond clearly, directly and confidently. Avoid reflections and answer requests without rambling."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=500,
        temperature = temperatureModel,
        top_p=0.85
    )

    # 📌 Debug: Print the raw response to analyze any problems
    print("DEBUG - Raw answer :", response)

    # 📌 Extract response from JSON
    chatbot_reply = response["choices"][0].get("message", {}).get("content", "").strip()

    # 📌 Save the conversation
    save_to_history(user_input, chatbot_reply)

    # 📌 Print the bot's response
    print("\n🤖 Bot:", chatbot_reply if chatbot_reply else "No response generated.")
