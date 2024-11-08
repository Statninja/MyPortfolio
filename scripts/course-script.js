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
