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
                document.getElementById('cropPredictionResult').textContent = JSON.stringify(result, null, 2);
            } catch (error) {
                document.getElementById('cropPredictionResult').textContent = 'Error: ' + error.message;
            }
        });
    }

    if (fertilizerRecommendationForm) {
        fertilizerRecommendationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(fertilizerRecommendationForm);
            const payload = {
                Temparature: parseFloat(formData.get('temperature_fr')),
                Humidity: parseFloat(formData.get('humidity_fr')),
                Moisture: parseFloat(formData.get('moisture_fr')),
                Soil_Type: formData.get('soil_type_fr'),
                Crop_Type: formData.get('crop_type_fr'),
                Nitrogen: parseFloat(formData.get('nitrogen_fr')),
                Potassium: parseFloat(formData.get('potassium_fr')),
                Phosphorous: parseFloat(formData.get('phosphorous_fr')),
            };
            try {
                const response = await fetch(`${API_URL}/fertilizerReccommendation`, { // Corrected endpoint name
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                document.getElementById('fertilizerRecommendationResult').textContent = JSON.stringify(result, null, 2);
            } catch (error) {
                document.getElementById('fertilizerRecommendationResult').textContent = 'Error: ' + error.message;
            }
        });
    }

    if (soilClassificationForm) {
        soilClassificationForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(soilClassificationForm);
            // 'file' is the expected key for the backend
            const payload = new FormData();
            payload.append('file', formData.get('soilImage'));

            try {
                const response = await fetch(`${API_URL}/soil_classification`, {
                    method: 'POST',
                    body: payload // FormData is sent directly, no need for Content-Type header for multipart/form-data
                });
                const result = await response.json();
                document.getElementById('soilClassificationResult').textContent = JSON.stringify(result, null, 2);
            } catch (error) {
                document.getElementById('soilClassificationResult').textContent = 'Error: ' + error.message;
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