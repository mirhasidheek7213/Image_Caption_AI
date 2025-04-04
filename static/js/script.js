document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.querySelector('.upload-box');
    const uploadBtn = document.getElementById('upload-btn');
    const imageInput = document.getElementById('image-input');
    const form = document.getElementById('upload-form');
    const submitBtn = document.getElementById('submit-btn');

    // Trigger file input click when upload button is clicked
    uploadBtn.addEventListener('click', () => {
        imageInput.click();
    });

    // Handle file selection
    imageInput.addEventListener('change', () => {
        if (imageInput.files.length > 0) {
            submitBtn.click(); // Automatically submit the form
        }
    });

    // Drag and drop functionality
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.backgroundColor = '#e0e0e0';
    });

    uploadBox.addEventListener('dragleave', () => {
        uploadBox.style.backgroundColor = '#f9f9f9';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.backgroundColor = '#f9f9f9';
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            imageInput.files = files;
            submitBtn.click(); // Automatically submit the form
        }
    });
});