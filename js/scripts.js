const API_URL = 'https://crop-prediction-1-6nny.onrender.com';

document.addEventListener('DOMContentLoaded', () => {
    const cropPredictionForm = document.getElementById('cropPredictionForm');
    const fertilizerRecommendationForm = document.getElementById('fertilizerRecommendationForm');
    const soilClassificationForm = document.getElementById('soilClassificationForm');

    if (cropPredictionForm) {
        cropPredictionForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(cropPredictionForm);
            const payload = {
                N: parseFloat(formData.get('N')),
                P: parseFloat(formData.get('P')),
                K: parseFloat(formData.get('K')),
                temperature: parseFloat(formData.get('temperature')),
                humidity: parseFloat(formData.get('humidity')),
                ph: parseFloat(formData.get('ph')),
                rainfall: parseFloat(formData.get('rainfall')),
            };
            try {
                const response = await fetch(`${API_URL}/prediction`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                const resultDiv = document.getElementById('cropPredictionResult');
                resultDiv.innerHTML = ''; // Clear previous results
                if (result.error) {
                    resultDiv.innerHTML = `<p class="error-message">Error: ${result.error}</p>`;
                } else if (result.top_prediction) {
                    let outputHtml = '<h3>Top Crop Predictions:</h3><ul>';
                    result.top_prediction.forEach(pred => {
                        outputHtml += `<li><strong>${pred.class}:</strong> ${(pred.probability * 100).toFixed(2)}%</li>`;
                    });
                    outputHtml += '</ul>';
                    resultDiv.innerHTML = outputHtml;
                } else {
                    resultDiv.innerHTML = '<p>No prediction data found.</p>';
                }
            } catch (error) {
                document.getElementById('cropPredictionResult').innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
            }
        });
    }

    if (fertilizerRecommendationForm) {
        fertilizerRecommendationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(fertilizerRecommendationForm);
            const payload = {
                Temparature: parseFloat(formData.get('temperature_fr')),
                Humidity: parseFloat(formData.get('humidity_fr')), // Corrected key based on Python API
                Moisture: parseFloat(formData.get('moisture_fr')),
                Soil_Type: formData.get('soil_type_fr'), // Corrected key based on Python API
                Crop_Type: formData.get('crop_type_fr'), // Corrected key based on Python API
                Nitrogen: parseFloat(formData.get('nitrogen_fr')),
                Potassium: parseFloat(formData.get('potassium_fr')),
                Phosphorous: parseFloat(formData.get('phosphorous_fr')),
            };
            try {
                const response = await fetch(`${API_URL}/fertilizerReccommendation`, { // Note: API endpoint is fertilizerReccommendation (double c)
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                const resultDiv = document.getElementById('fertilizerRecommendationResult');
                resultDiv.innerHTML = ''; // Clear previous results
                if (result.error) {
                    resultDiv.innerHTML = `<p class="error-message">Error: ${result.error}</p>`;
                } else if (result.prediction) {
                    resultDiv.innerHTML = `<h3>Recommended Fertilizer:</h3><p class="success-message">${result.prediction}</p>`;
                } else {
                    resultDiv.innerHTML = '<p>No recommendation data found.</p>';
                }
            } catch (error) {
                document.getElementById('fertilizerRecommendationResult').innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
            }
        });
    }

    if (soilClassificationForm) {
        soilClassificationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(soilClassificationForm);
            const payload = new FormData();
            payload.append('file', formData.get('soilImage'));

            try {
                const response = await fetch(`${API_URL}/soil_classification`, {
                    method: 'POST',
                    body: payload // FormData is sent directly, no need for Content-Type header for multipart/form-data
                });
                const result = await response.json();
                const resultDiv = document.getElementById('soilClassificationResult');
                resultDiv.innerHTML = ''; // Clear previous results
                if (result.error) {
                    resultDiv.innerHTML = `<p class="error-message">Error: ${result.error}</p>`;
                } else if (result.soil) {
                    resultDiv.innerHTML = `<h3>Soil Classification:</h3><p class="success-message">${result.soil}</p>`;
                } else {
                    resultDiv.innerHTML = '<p>No classification data found.</p>';
                }
            } catch (error) {
                document.getElementById('soilClassificationResult').innerHTML = `<p class="error-message">Error: ${error.message}</p>`;
            }
        });
    }

    // Active link highlighting
    const navLinks = document.querySelectorAll('nav ul li a');
    const currentPath = window.location.pathname.split('/').pop();
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
});