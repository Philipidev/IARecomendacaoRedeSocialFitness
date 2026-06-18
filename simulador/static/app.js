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

  // Métricas de avaliação offline do artigo (ranqueamento completo do catálogo),
  // por escala da base e por modelo. ndcg100 = NDCG@100; tema100 = acerto
  // temático@100 (item recomendado compartilha tema com o consumido depois,
  // Jaccard das tags >= 0,5). São valores precomputados no benchmark — o
  // simulador não os recalcula, apenas exibe como referência do modelo.
  const BENCH_METRICS = {
    "sf0.1": {
      popularity: { ndcg100: 0.1812, tema100: 1 },
      baseline_padrao: { ndcg100: 0.1057, tema100: 1 },
      baseline_otimizado: { ndcg100: 0.0507, tema100: 1 },
      ltr: { ndcg100: 0.1732, tema100: 1 },
    },
    sf3: {
      popularity: { ndcg100: 0.0003, tema100: 0.157 },
      baseline_padrao: { ndcg100: 0.0035, tema100: 0.103 },
      baseline_otimizado: { ndcg100: 0.0032, tema100: 0.662 },
      ltr: { ndcg100: 0.0082, tema100: 0.5 },
    },
    sf30: {
      popularity: { ndcg100: 0, tema100: 0.666 },
      baseline_padrao: { ndcg100: 0.0004, tema100: 0.116 },
      baseline_otimizado: { ndcg100: 0.0002, tema100: 0.083 },
      ltr: { ndcg100: 0, tema100: 0.773 },
    },
  };

  // Resolve a "variante" de métrica a partir da família + id do modelo, para
  // ser robusto a sufixos de versão (ex.: ltr_lightgbm_v1_core / _robusto).
  function benchVariant(model) {
    if (model.family === "popularity") return "popularity";
    if (model.family === "ltr_lightgbm") return "ltr";
    if (model.family === "baseline_hibrido") {
      return /otimiz/i.test(model.model_id)
        ? "baseline_otimizado"
        : "baseline_padrao";
    }
    return null;
  }

  function getBenchMetrics(model) {
    const scale = (model.scale_factor || "").toLowerCase();
    const variant = benchVariant(model);
    const porEscala = BENCH_METRICS[scale];
    if (!porEscala || !variant) return null;
    return porEscala[variant] || null;
  }

  // Formata número no padrão pt-BR com casas fixas (ex.: 0,0004).
  function fmtMetric(value, casas) {
    return Number(value).toFixed(casas).replace(".", ",");
  }

  // Exemplos pré-prontos para demonstração/print do TCC. Apontam para o
  // baseline híbrido padrão em sf30 (catálogo grande, recomendações coerentes).
  // O caso "mix" inclui tags fora do vocabulário (Pizza/JavaScript) que o
  // modelo ignora — demonstra robustez sem degradar o resultado fitness.
  const USE_CASES = {
    fitness_puro: {
      scale_factor: "sf30",
      model_id: "baseline_hibrido_padrao",
      tags: ["The_New_Workout_Plan", "Muscle_of_Love", "The_Weight"],
      top_k: 10,
      user_id: null,
      excluir_tags_exatas: false,
    },
    fitness_mix: {
      scale_factor: "sf30",
      model_id: "baseline_hibrido_padrao",
      tags: [
        "The_New_Workout_Plan",
        "Muscle_of_Love",
        "The_Weight",
        "Pizza",
        "JavaScript",
      ],
      top_k: 10,
      user_id: null,
      excluir_tags_exatas: false,
    },
    // Demonstra novidade + recência + relevância ao mesmo tempo: a maioria dos
    // posts recomendados traz uma tag escolhida (verde = relevância) somada a uma
    // tag relacionada nova (ex.: The_Weight = novidade), com datas recentes.
    fitness_descoberta: {
      scale_factor: "sf30",
      model_id: "baseline_hibrido_padrao",
      tags: ["The_New_Workout_Plan", "Carry_That_Weight", "Bicycle_Race"],
      top_k: 10,
      user_id: null,
      excluir_tags_exatas: false,
    },
  };

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
        opt.textContent = `${m.model_id} — ${family}`;
        opt.dataset.scaleFactor = m.scale_factor || "";
        opt.dataset.family = m.family;
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
    const bench = getBenchMetrics(m);
    const benchHtml = bench
      ? `<div class="model-metrics" title="Métricas da avaliação offline do artigo (ranqueamento completo do catálogo). Acerto temático = item recomendado compartilha tema com o consumido depois, com Jaccard das tags >= 0,5.">` +
        `<span class="metrics-title">📊 Avaliação offline (${m.scale_factor.toUpperCase()})</span>` +
        `<span class="metric-chip">NDCG@100: <strong>${fmtMetric(bench.ndcg100, 4)}</strong></span>` +
        `<span class="metric-chip">Acerto temático@100: <strong>${fmtMetric(bench.tema100, 3)}</strong></span>` +
        `</div>`
      : "";

    els.modelInfo.innerHTML =
      `<div><strong>${family}</strong> · ${m.n_posts.toLocaleString("pt-BR")} posts no catálogo · ` +
      `${m.n_tags} tags · dataset <code>${m.dataset_key}</code></div>` +
      (m.descricao ? `<em>${escapeHtml(m.descricao)}</em>` : "") +
      benchHtml;

    // preserva as tags selecionadas ao trocar de modelo; serão reaplicadas
    // sobre o vocabulário do novo modelo (tags fora do vocabulário viram chips
    // "oov", mantendo a seleção do usuário).
    const tagsPrevias = [...state.selectedTags];
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
      if (tagsPrevias.length > 0) selectTags(tagsPrevias);
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
      // chips fora do vocabulário (sintéticos) somem ao desmarcar
      if (chip.dataset.oov === "1") chip.remove();
    } else {
      state.selectedTags.add(tag);
      chip.classList.add("selected");
    }
    updateTagsSummary();
    updateRecommendButton();
  }

  function chipByTag() {
    const map = new Map();
    for (const chip of els.tagsContainer.querySelectorAll(".tag-chip")) {
      map.set(chip.dataset.tag, chip);
    }
    return map;
  }

  // Marca um conjunto de tags. Tags que existem no vocabulário marcam o chip
  // correspondente; tags fora do vocabulário viram um chip sintético "oov"
  // (visível e selecionado) para aparecerem no print e serem enviadas na query.
  function selectTags(tags) {
    clearAllTags();
    const existentes = chipByTag();
    for (const tag of tags) {
      const chip = existentes.get(tag);
      if (chip) {
        chip.classList.add("selected");
      } else {
        const oovChip = document.createElement("button");
        oovChip.type = "button";
        oovChip.className = "tag-chip oov selected";
        oovChip.dataset.tag = tag;
        oovChip.dataset.oov = "1";
        oovChip.textContent = tag;
        oovChip.title = "Tag fora do vocabulário do modelo (será ignorada na recomendação)";
        oovChip.addEventListener("click", () => toggleTag(tag, oovChip));
        els.tagsContainer.appendChild(oovChip);
      }
      state.selectedTags.add(tag);
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
      if (chip.dataset.oov === "1") {
        chip.remove();
      } else {
        chip.classList.remove("selected");
      }
    }
    updateTagsSummary();
    updateRecommendButton();
  }

  // Aplica um exemplo pré-pronto: seleciona o modelo, carrega suas tags e
  // preenche tags/opções. NÃO dispara a recomendação (o usuário clica depois).
  async function applyUseCase(caseId) {
    const uc = USE_CASES[caseId];
    if (!uc) return;

    const alvo = state.models.find(
      (m) => m.scale_factor === uc.scale_factor && m.model_id === uc.model_id,
    );
    if (!alvo) {
      showToast(
        `Modelo do exemplo indisponível (${uc.scale_factor} / ${uc.model_id}). Treine-o ou escolha outro.`,
      );
      return;
    }

    if (els.modelSelect.value !== alvo.model_dir) {
      els.modelSelect.value = alvo.model_dir;
      await onModelChange();
    }
    if (!state.tagsInfo) {
      showToast("Não foi possível carregar as tags do modelo do exemplo.");
      return;
    }

    els.topK.value = String(uc.top_k);
    els.userId.value = uc.user_id == null ? "" : String(uc.user_id);
    els.excluirExatas.checked = !!uc.excluir_tags_exatas;
    selectTags(uc.tags);
    showToast("Exemplo carregado. Clique em Recomendar.", "info");
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

  for (const btn of document.querySelectorAll(".use-case-btn")) {
    btn.addEventListener("click", () => applyUseCase(btn.dataset.case));
  }

  loadModels();
})();
