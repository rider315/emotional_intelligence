import { HfInference } from "@huggingface/inference";
import axios from "axios";
import express from "express";
import cors from "cors";
import "dotenv/config";

const app = express();

app.use(cors());
app.use(express.json());

// Endpoint to get advice
app.post("/predict", async (req, res) => {
  const { userInput } = req.body;
  if (!userInput) {
    return res.status(400).json({
      success: false,
      error: "No input provided",
    });
  }

  try {
    const detectedEmotion = await detectEmotion(userInput);
    const adviceResponse = await generateAdvice(userInput);
    res.json({
      success: true,
      detectedEmotion,
      adviceResponse,
    });
  } catch (error) {
    console.error("Error in /predict:", error);
    res.status(500).json({
      success: false,
      error: "Sorry, there was an error. Please try again later.",
    });
  }
});

async function detectEmotion(inputText) {
  try {
    const payload = { userInput: inputText };
    const response = await axios.post(
      "http://localhost:3000/predict", // Flask backend runs on port 3000
      payload,
      { headers: { "Content-Type": "application/json" } }
    );
    // Extract and return the detected emotion
    const detectedEmotion = response.data.detectedEmotion || "Unknown";
    return detectedEmotion;
  } catch (error) {
    console.error("Error detecting emotion:", error.message);
    return "Unknown";
  }
}

const hf = new HfInference(process.env.HF_API_KEY);

async function generateAdvice(userInput) {
  const prompt = `You are a mental health advisor. The user's input is: "${userInput}". Please provide empathetic advice in clear bullet points, including coping mechanisms, self-care strategies, and additional resources for support. Use clear language, avoiding medical jargon, and ensure the response is a bulleted list. Conclude with a statement encouraging the user to seek additional help if needed. Limit words to 200.`;

  try {
    const response = await hf.textGeneration({
      model: "SuyashPandey/finetuned-mental-health-advisor",
      inputs: prompt,
      parameters: {
        max_new_tokens: 200,
        temperature: 0.7,
        top_k: 50,
        top_p: 0.95,
      },
    });

    console.log("Model Response:", response.generated_text);
    return response.generated_text;
  } catch (error) {
    console.error("Error fetching advice:", error);
    return "Unable to generate advice at the moment. Please try again later.";
  }
}

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});