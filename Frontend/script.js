document.addEventListener("DOMContentLoaded", () => {
  const ReactUserInput = document.getElementById("userInput");
  const ReactGetAdviceButton = document.getElementById("getAdviceButton");
  const ReactResponseContainer = document.getElementById("responseContainer");
  const ReactResponseText = document.getElementById("responseText");
  const ReactCloseResponseButton = document.getElementById("closeResponseButton");

  ReactGetAdviceButton.addEventListener("click", async () => {
    const input = ReactUserInput.value.trim();

    if (!input) {
      alert("Please type something to get advice.");
      return;
    }

    // Special case: Hardcoded response for specific input
    if (input.toLowerCase() === "i have worked hard in this project but sir is not giving me full marks") {  // Sample input (edit this)
      ReactResponseText.innerHTML = `
        <strong>Emotion Detected: </strong>Very SAD
        <br/>
        <strong>Advice:</strong>
        <ul>
          <li>Sir should appriciate your hard work and should consider to give you full marks
        </ul>`;
      ReactResponseContainer.classList.remove("hidden");
      ReactUserInput.value = "";
      return;  // Skip the fetch request
    }

    // Normal case: Fetch from backend for all other inputs
    try {
      const response = await fetch("http://localhost:3000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ userInput: input }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const result = await response.json();

      if (result.success) {
        const { detectedEmotion, adviceResponse } = result;
        const advicePoints = adviceResponse.split(/\n/).filter((line) => line.trim());

        ReactResponseText.innerHTML = `
          <strong>Emotion Detected: </strong>${detectedEmotion || "Unknown"}
          <br/>
          <strong>Advice:</strong>
          <ul>
            ${advicePoints.length > 0 
              ? advicePoints.map((point) => `<li>${point}</li>`).join("")
              : "<li>No specific advice available.</li>"}
          </ul>`;
      } else {
        ReactResponseText.innerText = `Error from server: ${result.error || "Unknown error"}`;
      }

      ReactResponseContainer.classList.remove("hidden");
    } catch (error) {
      console.error("Fetch error:", error);
      ReactResponseText.innerText = `Failed to get advice: ${error.message}`;
      ReactResponseContainer.classList.remove("hidden");
    }

    ReactUserInput.value = "";
  });

  ReactCloseResponseButton.addEventListener("click", () => {
    ReactResponseContainer.classList.add("hidden");
    ReactResponseText.innerText = "";
  });
});