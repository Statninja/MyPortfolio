// scripts/course-script.js

document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute("href")).scrollIntoView({
                behavior: "smooth"
            });
        });
    });
});

// scripts/course-script.js pyodide

let pyodide;

async function initializePyodide() {
    pyodide = await loadPyodide();
    console.log("Pyodide loaded successfully.");
}

document.addEventListener("DOMContentLoaded", initializePyodide);

// Chatbot interaction
function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    const chatbox = document.getElementById("chatbox");

    // Check if input is empty
    if (!userInput.trim()) return;

    // Display the user's message
    const userMessage = document.createElement("p");
    userMessage.classList.add("user-message");
    userMessage.textContent = "User: " + userInput;
    chatbox.appendChild(userMessage);

    // Clear the input field after sending
    document.getElementById("userInput").value = '';

    // Simulated response (for demonstration purposes)
    const botMessage = document.createElement("p");
    botMessage.classList.add("bot-message");
    botMessage.textContent = "Bot: Hello! How can I help you?";
    chatbox.appendChild(botMessage);

    // Auto-scroll to the latest message
    chatbox.scrollTop = chatbox.scrollHeight;
}
