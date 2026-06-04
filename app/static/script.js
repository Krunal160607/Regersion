document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector(".prediction-form");
    const resultBox = document.querySelector("#result");
    const errorBox = document.querySelector("#error");
    const countrySelect = document.querySelector('select[name="country_name"]');
    const countryLabelBadge = document.querySelector("#country-label-badge");
    const countryLabelPreview = document.querySelector("#country-label-preview");
    const currencyFormatter = new Intl.NumberFormat(undefined, {
        style: "currency",
        currency: "USD",
        maximumFractionDigits: 2,
    });

    if (!form) {
        return;
    }

    const syncCountryMeta = () => {
        const selected = countrySelect?.selectedOptions?.[0];
        const label = selected?.dataset?.label || "N/A";

        if (countryLabelBadge) {
            countryLabelBadge.textContent = `Label ${label}`;
        }

        if (countryLabelPreview) {
            countryLabelPreview.textContent = label;
        }
    };

    syncCountryMeta();
    countrySelect?.addEventListener("change", syncCountryMeta);

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector("button");
        const formData = new FormData(form);
        const payload = {
            country_name: formData.get("country_name"),
            year: Number(formData.get("year")),
        };

        if (resultBox) {
            resultBox.classList.add("hidden");
        }

        if (errorBox) {
            errorBox.classList.add("hidden");
        }

        if (button) {
            button.disabled = true;
            button.textContent = "Predicting...";
        }

        try {
            const response = await fetch(form.dataset.apiUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(payload),
            });
            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || "Prediction failed.");
            }

            if (resultBox) {
                resultBox.innerHTML = `
                    <div class="result-eyebrow">Prediction Ready</div>
                    <div class="result-value">${currencyFormatter.format(data.predicted_gdp)}</div>
                    <div class="result-meta">
                        ${data.country_name} · Label ${data.country_label} · ${data.country_code || "No code"}
                    </div>
                    <div class="result-copy">
                        Forecast generated for ${data.year} using the notebook-aligned decision tree model.
                    </div>
                `;
                resultBox.classList.remove("hidden");
            }
        } catch (error) {
            if (errorBox) {
                errorBox.textContent = error.message;
                errorBox.classList.remove("hidden");
            }
        } finally {
            if (button) {
                button.disabled = false;
                button.textContent = "Predict GDP";
            }
        }
    });
});
