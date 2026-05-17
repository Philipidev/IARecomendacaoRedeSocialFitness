(() => {
  "use strict";

  const els = {
    modelSelect: document.getElementById("model-select"),
    modelInfo: document.getElementById("model-info"),
    tagSearch: document.getElementById("tag-search"),
    clearTags: document.getElementById("clear-tags"),
    tagsContainer: document.getElementById("tags-container"),
    tagsSummary: document.getElementById("tags-summary"),
    topK: document.getElementById("top-k"),
    userId: document.getElementById("user-id"),
    excluirExatas: document.getElementById("excluir-exatas"),
    recommendBtn: document.getElementById("recommend-btn"),
    resultsContainer: document.getElementById("results-container"),
    resultsMeta: document.getElementById("results-meta"),
    toast: document.getElementById("toast"),
  };

  const FAMILY_LABEL = {
    popularity: "Popularidade",
    baseline_hibrido: "Híbrido",
    ltr_lightgbm: "LTR (LightGBM)",
  };

  // Melhor modelo por dataset segundo o benchmark TCC (resumo_benchmark_3_versoes.md):
  // - SF30: ltr_lightgbm_v1_robusto (NDCG@10 = 0.0009, melhor Recall@10)
  // - SF3:  ltr_lightgbm_v1_core    (NDCG@10 = 0.0008, melhor Recall@10)
  // - SF0.1: sem queries válidas no benchmark — sem destaque.
  const BEST_BY_SCALE = {
    sf30: "ltr_lightgbm_v1_robusto",
    sf3: "ltr_lightgbm_v1_core",
  };
  const OVERALL_BEST = { scale_factor: "sf30", model_id: "ltr_lightgbm_v1_robusto" };

  function isBest(model) {
    return BEST_BY_SCALE[model.scale_factor] === model.model_id;
  }
  function isOverallBest(model) {
    return (
      model.scale_factor === OVERALL_BEST.scale_factor &&
      model.model_id === OVERALL_BEST.model_id
    );
  }

  const state = {
    models: [],
    selectedModel: null,
    tags: [],
    selectedTags: new Set(),
    tagsInfo: null,
  };

  function showToast(message, kind = "error") {
    els.toast.textContent = message;
    els.toast.className = `toast ${kind === "error" ? "" : kind}`;
    els.toast.hidden = false;
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => (els.toast.hidden = true), 4000);
  }

  async function fetchJson(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let detail;
      try {
        const payload = await res.json();
        detail = payload.detail || JSON.stringify(payload);
      } catch (_) {
        detail = await res.text();
      }
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return res.json();
  }

  async function loadModels() {
    try {
      const data = await fetchJson("/api/models");
      state.models = data.models || [];
      renderModels();
    } catch (err) {
      showToast(`Erro ao carregar modelos: ${err.message}`);
      els.modelSelect.innerHTML = '<option>Erro ao carregar modelos</option>';
    }
  }

  function renderModels() {
    if (state.models.length === 0) {
      els.modelSelect.innerHTML = '<option>Nenhum modelo encontrado</option>';
      els.modelSelect.disabled = true;
      return;
    }

    // agrupa por scale_factor / dataset_key
    const byDataset = new Map();
    for (const m of state.models) {
      const key = m.scale_factor || m.dataset_key || "desconhecido";
      if (!byDataset.has(key)) byDataset.set(key, []);
      byDataset.get(key).push(m);
    }

    const frag = document.createDocumentFragment();
    for (const [scale, models] of byDataset.entries()) {
      const og = document.createElement("optgroup");
      og.label = `${scale.toUpperCase()} (${models[0].dataset_key})`;
      for (const m of models) {
        const opt = document.createElement("option");
        opt.value = m.model_dir;
        const family = FAMILY_LABEL[m.family] || m.family;
        let prefix = "";
        if (isOverallBest(m)) prefix = "🏆 ";
        else if (isBest(m)) prefix = "⭐ ";
        opt.textContent = `${prefix}${m.model_id} — ${family}`;
        opt.dataset.scaleFactor = m.scale_factor || "";
        opt.dataset.family = m.family;
        if (isBest(m)) opt.classList.add("best-option");
        og.appendChild(opt);
      }
      frag.appendChild(og);
    }

    els.modelSelect.innerHTML = "";
    els.modelSelect.appendChild(frag);
    els.modelSelect.disabled = false;

    // seleciona um padrão razoável: maior scale + LTR robusto, se houver
    const preferred =
      state.models.find(
        (m) =>
          m.scale_factor === "sf30" && m.model_id === "ltr_lightgbm_v1_robusto",
      ) ||
      state.models.find((m) => m.scale_factor === "sf30") ||
      state.models[state.models.length - 1];

    els.modelSelect.value = preferred.model_dir;
    onModelChange();
  }

  async function onModelChange() {
    const modelDir = els.modelSelect.value;
    state.selectedModel = state.models.find((m) => m.model_dir === modelDir);
    if (!state.selectedModel) return;

    const m = state.selectedModel;
    const family = FAMILY_LABEL[m.family] || m.family;
    let badge = "";
    if (isOverallBest(m)) {
      badge = '<span class="best-badge overall">🏆 Melhor modelo do benchmark TCC</span>';
    } else if (isBest(m)) {
      badge = `<span class="best-badge">⭐ Melhor modelo em ${m.scale_factor.toUpperCase()}</span>`;
    }
    els.modelInfo.innerHTML =
      badge +
      `<div><strong>${family}</strong> · ${m.n_posts.toLocaleString("pt-BR")} posts no catálogo · ` +
      `${m.n_tags} tags · dataset <code>${m.dataset_key}</code></div>` +
      (m.descricao ? `<em>${escapeHtml(m.descricao)}</em>` : "");

    // limpa seleção ao trocar modelo (vocabulários podem diferir)
    state.selectedTags = new Set();
    els.tagsContainer.innerHTML = '<p class="placeholder">Carregando tags…</p>';
    els.recommendBtn.disabled = true;

    try {
      const data = await fetchJson(
        `/api/tags?model_dir=${encodeURIComponent(modelDir)}`,
      );
      state.tagsInfo = data;
      state.tags = data.tags || [];
      renderTags();
      updateRecommendButton();
    } catch (err) {
      showToast(`Erro ao carregar tags: ${err.message}`);
      els.tagsContainer.innerHTML =
        '<p class="placeholder">Não foi possível carregar as tags.</p>';
    }
  }

  function renderTags() {
    if (state.tags.length === 0) {
      els.tagsContainer.innerHTML =
        '<p class="placeholder">Modelo sem tags conhecidas.</p>';
      return;
    }
    const frag = document.createDocumentFragment();
    for (const tag of state.tags) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "tag-chip";
      chip.dataset.tag = tag;
      chip.textContent = tag;
      if (state.selectedTags.has(tag)) chip.classList.add("selected");
      chip.addEventListener("click", () => toggleTag(tag, chip));
      frag.appendChild(chip);
    }
    els.tagsContainer.innerHTML = "";
    els.tagsContainer.appendChild(frag);
    applyTagFilter();
    updateTagsSummary();
  }

  function toggleTag(tag, chip) {
    if (state.selectedTags.has(tag)) {
      state.selectedTags.delete(tag);
      chip.classList.remove("selected");
    } else {
      state.selectedTags.add(tag);
      chip.classList.add("selected");
    }
    updateTagsSummary();
    updateRecommendButton();
  }

  function applyTagFilter() {
    const query = els.tagSearch.value.trim().toLowerCase();
    for (const chip of els.tagsContainer.querySelectorAll(".tag-chip")) {
      const tag = chip.dataset.tag.toLowerCase();
      chip.classList.toggle("hidden", query !== "" && !tag.includes(query));
    }
  }

  function updateTagsSummary() {
    const n = state.selectedTags.size;
    if (n === 0) {
      els.tagsSummary.textContent = "Nenhuma tag selecionada.";
    } else {
      els.tagsSummary.textContent = `${n} tag${n === 1 ? "" : "s"} selecionada${n === 1 ? "" : "s"}: ${[...state.selectedTags].join(", ")}`;
    }
  }

  function updateRecommendButton() {
    if (!state.selectedModel) {
      els.recommendBtn.disabled = true;
      return;
    }
    // PopularityRanker ignora as tags — permite recomendar sem nenhuma selecionada.
    const requerTags = state.selectedModel.family !== "popularity";
    els.recommendBtn.disabled = requerTags && state.selectedTags.size === 0;
  }

  function clearAllTags() {
    state.selectedTags.clear();
    for (const chip of els.tagsContainer.querySelectorAll(".tag-chip")) {
      chip.classList.remove("selected");
    }
    updateTagsSummary();
    updateRecommendButton();
  }

  async function runRecommend() {
    if (!state.selectedModel || state.selectedTags.size === 0) return;
    els.recommendBtn.disabled = true;
    els.resultsContainer.innerHTML =
      '<div class="loading-state"><span class="spinner"></span>Calculando recomendações…</div>';
    els.resultsMeta.textContent = "";

    const userIdRaw = els.userId.value.trim();
    const body = {
      model_dir: state.selectedModel.model_dir,
      tags: [...state.selectedTags],
      top_k: parseInt(els.topK.value, 10),
      user_id: userIdRaw ? parseInt(userIdRaw, 10) : null,
      excluir_tags_exatas: els.excluirExatas.checked,
    };

    try {
      const data = await fetchJson("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      renderResults(data);
    } catch (err) {
      showToast(`Erro ao recomendar: ${err.message}`);
      els.resultsContainer.innerHTML =
        '<p class="placeholder">Não foi possível gerar recomendações.</p>';
    } finally {
      updateRecommendButton();
    }
  }

  function renderResults(data) {
    const { items = [], meta = {} } = data;
    if (meta.tags_desconhecidas && meta.tags_desconhecidas.length > 0) {
      showToast(
        `Tags ignoradas (fora do vocabulário): ${meta.tags_desconhecidas.join(", ")}`,
        "warn",
      );
    }

    const parts = [
      `${items.length} de ${meta.top_k || items.length}`,
      `modelo ${meta.model_id || "?"}`,
    ];
    if (meta.scale_factor) parts.push(meta.scale_factor.toUpperCase());
    if (meta.personalizado) parts.push("personalizado");
    if (meta.timestamp_usado)
      parts.push(`ts ref: ${new Date(meta.timestamp_usado).toLocaleDateString("pt-BR")}`);
    els.resultsMeta.textContent = parts.join(" · ");

    if (items.length === 0) {
      els.resultsContainer.innerHTML =
        '<p class="placeholder">Nenhum resultado para essas tags.</p>';
      return;
    }

    const list = document.createElement("div");
    list.className = "results-list";
    items.forEach((item, idx) => {
      list.appendChild(renderCard(item, idx + 1));
    });
    els.resultsContainer.innerHTML = "";
    els.resultsContainer.appendChild(list);
  }

  function renderCard(item, rank) {
    const card = document.createElement("article");
    card.className = "result-card";

    const rankEl = document.createElement("div");
    rankEl.className = "result-rank";
    rankEl.textContent = `#${rank}`;

    const main = document.createElement("div");
    main.className = "result-main";

    const metaRow = document.createElement("div");
    metaRow.className = "result-meta-row";

    const typeBadge = document.createElement("span");
    const type = (item.message_type || "?").toLowerCase();
    typeBadge.className = `message-type-icon ${type}`;
    typeBadge.textContent = type;
    metaRow.appendChild(typeBadge);

    const date = formatDate(item.creation_date_iso);
    if (date) metaRow.appendChild(textSpan(`📅 ${date}`));
    if (item.language) metaRow.appendChild(textSpan(`🌐 ${item.language}`));
    if (item.content_length != null)
      metaRow.appendChild(textSpan(`📝 ${item.content_length} chars`));

    const tags = document.createElement("div");
    tags.className = "result-tags";
    for (const tag of item.tags_fitness || []) {
      const badge = document.createElement("span");
      badge.className = "tag-badge";
      if (state.selectedTags.has(tag)) badge.classList.add("match");
      badge.textContent = tag;
      tags.appendChild(badge);
    }

    main.appendChild(metaRow);
    main.appendChild(tags);

    const scoreWrap = document.createElement("div");
    scoreWrap.className = "result-score";
    const score = Number(item.relevance_score || 0);
    const scoreVal = document.createElement("span");
    scoreVal.className = "score-value";
    scoreVal.textContent = score.toFixed(4);
    const bar = document.createElement("div");
    bar.className = "score-bar";
    const fill = document.createElement("div");
    fill.className = "score-bar-fill";
    fill.style.width = `${Math.max(0, Math.min(1, score)) * 100}%`;
    bar.appendChild(fill);
    scoreWrap.appendChild(scoreVal);
    scoreWrap.appendChild(bar);

    card.appendChild(rankEl);
    card.appendChild(main);
    card.appendChild(scoreWrap);
    return card;
  }

  function textSpan(text) {
    const s = document.createElement("span");
    s.textContent = text;
    return s;
  }

  function formatDate(iso) {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      if (Number.isNaN(d.getTime())) return iso;
      return d.toLocaleDateString("pt-BR", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
      });
    } catch (_) {
      return iso;
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]),
    );
  }

  els.modelSelect.addEventListener("change", onModelChange);
  els.tagSearch.addEventListener("input", applyTagFilter);
  els.clearTags.addEventListener("click", clearAllTags);
  els.recommendBtn.addEventListener("click", runRecommend);

  loadModels();
})();
