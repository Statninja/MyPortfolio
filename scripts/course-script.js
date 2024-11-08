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
