const $ = (id) => document.getElementById(id);

async function loadPredictions() {
  const day = $("day")?.value || "today";
  const notice = $("notice");

  try {
    if (notice) {
      notice.textContent = "Running analysis...";
      notice.classList.remove("hidden");
    }

    const response = await fetch(`/api/predictions?day=${encodeURIComponent(day)}`);
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Unable to load predictions");
    }

    renderPredictions(data);
  } catch (error) {
    if (notice) {
      notice.textContent = `Error: ${error.message}`;
      notice.classList.remove("hidden");
    }
  }
}

function renderPredictions(data) {
  const hero = $("hero");
  const detail = $("detail");
  const notice = $("notice");

  if (notice) notice.classList.add("hidden");

  const matches = data.matches || data.predictions || [];

  if (!matches.length) {
    if (hero) {
      hero.innerHTML = `
        <div class="empty-state">
          <h2>No matches found</h2>
          <p>There are currently no analysed fixtures for this selection.</p>
        </div>`;
    }
    return;
  }

  const ranked = [...matches].sort((a, b) =>
    (b.confidence || b.model_probability || 0) -
    (a.confidence || a.model_probability || 0)
  );

  if (hero) {
    hero.innerHTML = ranked.map((m, index) => {
      const home = m.home || m.home_team || "Home";
      const away = m.away || m.away_team || "Away";
      const pick = m.pick || m.prediction || "—";
      const confidence =
        Number(m.confidence ?? m.model_probability ?? 0);

      const pct = confidence <= 1
        ? confidence * 100
        : confidence;

      const decision =
        String(m.decision || "PREDICTION ONLY").toUpperCase();

      return `
        <article class="match-card ${decision.toLowerCase().replaceAll(" ", "-")}">
          <div class="rank">#${index + 1}</div>
          <div class="fixture">
            <strong>${home}</strong>
            <span>v</span>
            <strong>${away}</strong>
          </div>
          <div class="pick">${pick}</div>
          <div class="confidence">${pct.toFixed(1)}%</div>
          <div class="decision">${decision}</div>
        </article>`;
    }).join("");
  }

  if (detail) {
    detail.innerHTML = `
      <div class="analysis-summary">
        <strong>${ranked.length}</strong> matches analysed
      </div>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const day = $("day");

  if (day) {
    day.addEventListener("change", loadPredictions);
  }

  loadPredictions();
});
