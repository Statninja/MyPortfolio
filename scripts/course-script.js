// Smooth scrolling for sidebar links
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll('#course-sidebar a').forEach(anchor => {
        anchor.addEventListener("click", function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute("href")).scrollIntoView({
                behavior: "smooth"
            });
        });
    });
});

// Initialize Pyodide
let pyodide;
async function initializePyodide() {
    // Load Pyodide
    pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.18.1/full/" });
    console.log("Pyodide loaded successfully.");

    // Load numpy package
    await pyodide.loadPackage("numpy");
    console.log("Numpy loaded successfully.");

    // Load the God Algorithm Python code
    const response = await fetch("python/god_algorithm.py");
    const code = await response.text();
    await pyodide.runPythonAsync(code);
    console.log("God Algorithm loaded successfully.");
}

// Run Pyodide initialization on page load
document.addEventListener("DOMContentLoaded", initializePyodide);

// Function to call `get_knowledge` from god_algorithm.py
async function searchKnowledge(keyword) {
    await pyodide.runPythonAsync(`
        result = get_knowledge("${keyword}")
    `);
    const result = pyodide.globals.get("result");
    return result.toJs();
}

// Function to display knowledge search results
async function displaySection() {
    const keyword = document.getElementById("sectionInput").value;
    const outputDiv = document.getElementById("output");

    // Clear previous results
    outputDiv.innerHTML = '';

    // Fetch results from the God Algorithm
    const results = await searchKnowledge(keyword);
    results.forEach(paragraph => {
        const paraElement = document.createElement('p');
        paraElement.textContent = paragraph;
        outputDiv.appendChild(paraElement);
    });
}

// Chatbot interaction function
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
