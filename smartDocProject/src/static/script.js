function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert("Please select a PDF file first.");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("pdf", file);

    fetch("/summarize", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("summaryText").innerText = data.summary;
        document.getElementById("classification").innerText = data.classification;
        document.getElementById("lengthInfo").innerText = `Original Length: ${data.original_length}, Summarized Length: ${data.summary_length}`;
        document.getElementById("summarySection").classList.remove("hidden");
    })
    .catch(error => console.error("Error:", error));
}

// script.js

// Handling sticky login button
document.querySelector('.btn-sticky-login').addEventListener('click', function() {
    alert('Welcome! You clicked the Login button.');
});
