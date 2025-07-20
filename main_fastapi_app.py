from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pandas as pd
import io
import base64

app = FastAPI(title="Housing Price Prediction API")

# Serve static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model (ensure the file exists!)
model = joblib.load("california_knn_model.pkl")

@app.post("/predict_csv/")
async def predict_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        expected_cols = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        if not all(col in df.columns for col in expected_cols):
            raise HTTPException(status_code=400, detail="Missing one or more required columns.")

        predictions = model.predict(df)
        df["PredictedPrice"] = predictions.round(4)
        preview = df.head(10).round(4).to_dict(orient="records")
        result_csv = df.to_csv(index=False)
        b64_csv = base64.b64encode(result_csv.encode()).decode()

        return {
            "message": "Prediction complete",
            "preview": preview,
            "csv_base64": b64_csv,
            "filename": "predictions.csv"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>California Housing Price Predictor</title>
        <link rel="stylesheet" href="/static/styles.css?v=1.0">
    </head>
    <body>
        <main>
            <h3>Bulk Prediction from CSV</h3>
            <form id="csvForm">
                <input type="file" id="csvFile" name="file" accept=".csv" required />
                <button type="submit">Upload and Predict</button>
            </form>
            <div id="csvPreviewContainer" style="margin-top: 20px; display: none;">
                <h4>Preview (first 10 rows):</h4>
                <table id="csvPreview" border="1" cellpadding="5"></table>
            </div>
            <a id="downloadLink" class = "hidden" download="predictions.csv">Download Full Predictions</a>
        </main>

        <script>
            const csvForm = document.getElementById('csvForm');
            const csvPreview = document.getElementById('csvPreview');
            const csvPreviewContainer = document.getElementById('csvPreviewContainer');
            const downloadLink = document.getElementById('downloadLink');

            function base64ToBlob(base64, mime) {
                const binary = atob(base64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                return new Blob([bytes], { type: mime });
            }

            csvForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('csvFile');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/predict_csv/', {
                        method: 'POST',
                        body: formData,
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        csvPreview.innerHTML = `<tr><td colspan="100%">${data.detail}</td></tr>`;
                        downloadLink.style.display = "none";
                        return;
                    }

                    const previewData = data.preview;
                    if (previewData.length > 0) {
                        const headers = Object.keys(previewData[0]);
                        csvPreview.innerHTML = `
                            <thead>
                                <tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>
                                </thead>
                            <tbody>
                                ${previewData.map(row =>
                                    `<tr>${headers.map(h => `<td>${row[h]}</td>`).join('')}</tr>`
                                ).join('')}
                            </tbody>`;
                        csvPreviewContainer.style.display = "block";
                    }

                    const blob = base64ToBlob(data.csv_base64, 'text/csv');
                    const url = URL.createObjectURL(blob);
                    downloadLink.href = url;
                    downloadLink.download = data.filename || "predictions.csv";
                    downloadLink.classList.remove("hidden");
                    downloadLink.classList.add("visible");
                } catch (error) {
                    console.error("Error:", error);
                    csvPreview.innerHTML = `<tr><td colspan="100%">Request failed: ${error.message}</td></tr>`;
                    downloadLink.style.display = "none";
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
