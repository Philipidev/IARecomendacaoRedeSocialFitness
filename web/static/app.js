/**
 * app.js — Logica do dashboard: SSE, acoes, atualizacao de DOM.
 */

let eventSource = null;

// ---------------------------------------------------------------------------
// SSE — Streaming de execucao
// ---------------------------------------------------------------------------

function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }

    const logOutput = document.getElementById('log-output');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const execStage = document.getElementById('exec-stage');
    const execStageLabel = document.getElementById('exec-stage-label');
    const execStageStep = document.getElementById('exec-stage-step');
    const execDetail = document.getElementById('exec-detail');
    const execStatusBadge = document.getElementById('exec-status-badge');
    const execTaskLabel = document.getElementById('exec-task-label');

    eventSource = new EventSource('/api/run/stream');

    eventSource.addEventListener('log', function(e) {
        if (logOutput) {
            logOutput.textContent += e.data + '\n';
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    });

    eventSource.addEventListener('progress', function(e) {
        const data = JSON.parse(e.data);
        if (progressContainer) progressContainer.style.display = 'block';
        if (progressBar) {
            progressBar.style.width = data.percent + '%';
            progressText.textContent = data.percent + '%';
        }
        if (execDetail) {
            execDetail.textContent = data.detail || '';
        }
    });

    eventSource.addEventListener('stage', function(e) {
        const data = JSON.parse(e.data);
        if (execStage) execStage.style.display = 'block';
        if (execStageLabel) execStageLabel.textContent = data.label;
        if (execStageStep) execStageStep.textContent = '(' + data.current + '/' + data.total + ')';
        // Reset barra de progresso por etapa
        if (progressBar) {
            progressBar.style.width = '0%';
            if (progressText) progressText.textContent = '0%';
        }
    });

    eventSource.addEventListener('task', function(e) {
        const data = JSON.parse(e.data);
        if (execTaskLabel) execTaskLabel.textContent = data.label;
    });

    eventSource.addEventListener('status', function(e) {
        const status = e.data;
        if (execStatusBadge) {
            execStatusBadge.textContent = status.toUpperCase();
            execStatusBadge.className = 'exec-badge ' + status;
        }

        if (status === 'completed' || status === 'error' || status === 'cancelled') {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            refreshDashboard();
        }
    });

    eventSource.addEventListener('state_updated', function() {
        refreshDashboard();
    });

    eventSource.onerror = function() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    };
}

// ---------------------------------------------------------------------------
// Acoes
// ---------------------------------------------------------------------------

function submitAction(event, url) {
    event.preventDefault();
    const form = event.target;
    const formData = new FormData(form);
    const body = {};
    for (const [key, value] of formData.entries()) {
        // Tentar converter para numero se possivel
        const num = Number(value);
        body[key] = isNaN(num) ? value : num;
    }

    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            showNotification(data.message || data.dataset || data.model || data.benchmark || 'OK', 'ok');
            refreshDashboard();
        } else {
            showNotification(data.detail || 'Erro desconhecido', 'error');
        }
    })
    .catch(err => showNotification('Erro: ' + err.message, 'error'));

    return false;
}

function submitEvaluation(event) {
    event.preventDefault();
    const form = event.target;
    const checked = form.querySelectorAll('input[name="modes"]:checked');
    const modes = Array.from(checked).map(cb => cb.value);

    if (modes.length === 0) {
        showNotification('Selecione pelo menos um modo de avaliacao.', 'error');
        return false;
    }

    // Expandir "all" — se selecionou "all", enviar so "all"
    const finalModes = modes.includes('all') ? ['all'] : modes;

    runActionWithBody('/api/run/evaluation', { modes: finalModes });
    return false;
}

function runAction(url, body) {
    runActionWithBody(url, body || {});
}

function runActionWithBody(url, body) {
    // Limpar console
    const logOutput = document.getElementById('log-output');
    if (logOutput) logOutput.textContent = '';

    const progressContainer = document.getElementById('progress-container');
    if (progressContainer) progressContainer.style.display = 'none';

    const execStage = document.getElementById('exec-stage');
    if (execStage) execStage.style.display = 'none';

    const execDetail = document.getElementById('exec-detail');
    if (execDetail) execDetail.textContent = '';

    fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    })
    .then(r => {
        if (r.status === 409) {
            showNotification('Ja existe uma execucao em andamento.', 'error');
            return null;
        }
        return r.json();
    })
    .then(data => {
        if (data && data.ok) {
            showNotification(data.message, 'ok');
            connectSSE();
            setButtonsDisabled(true);
        } else if (data) {
            showNotification(data.detail || 'Erro ao iniciar acao.', 'error');
        }
    })
    .catch(err => showNotification('Erro: ' + err.message, 'error'));
}

function cancelExecution() {
    fetch('/api/cancel', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
        showNotification(data.message || data.detail, data.ok ? 'ok' : 'error');
    })
    .catch(err => showNotification('Erro: ' + err.message, 'error'));
}

// ---------------------------------------------------------------------------
// Dashboard refresh
// ---------------------------------------------------------------------------

function refreshDashboard() {
    htmx.ajax('GET', '/partials/status-header', { target: '#status-header', swap: 'innerHTML' });
    htmx.ajax('GET', '/partials/status-cards', { target: '#status-cards', swap: 'innerHTML' });
    htmx.ajax('GET', '/partials/actions-panel', { target: '#actions-panel', swap: 'innerHTML' });
    htmx.ajax('GET', '/partials/execution-panel', { target: '#execution-panel', swap: 'innerHTML' });
    htmx.ajax('GET', '/partials/history-panel', { target: '#history-panel', swap: 'innerHTML' });
}

function setButtonsDisabled(disabled) {
    document.querySelectorAll('.actions-panel button, .actions-panel select').forEach(el => {
        el.disabled = disabled;
    });
}

// ---------------------------------------------------------------------------
// Notificacoes
// ---------------------------------------------------------------------------

function showNotification(message, type) {
    let container = document.getElementById('notification-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = 'position:fixed;top:16px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
        document.body.appendChild(container);
    }

    const el = document.createElement('div');
    el.style.cssText = 'padding:10px 18px;border-radius:6px;font-size:0.85rem;font-weight:500;opacity:0;transition:opacity 0.3s;max-width:400px;word-break:break-word;';
    if (type === 'ok') {
        el.style.background = 'rgba(52,211,153,0.15)';
        el.style.color = '#34d399';
        el.style.border = '1px solid rgba(52,211,153,0.3)';
    } else {
        el.style.background = 'rgba(248,113,113,0.15)';
        el.style.color = '#f87171';
        el.style.border = '1px solid rgba(248,113,113,0.3)';
    }
    el.textContent = message;
    container.appendChild(el);

    requestAnimationFrame(() => { el.style.opacity = '1'; });
    setTimeout(() => {
        el.style.opacity = '0';
        setTimeout(() => el.remove(), 300);
    }, 4000);
}

// ---------------------------------------------------------------------------
// Auto-conectar SSE se houver execucao ativa
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/execution/status')
    .then(r => r.json())
    .then(data => {
        if (data.status === 'running') {
            connectSSE();
        }
    })
    .catch(() => {});
});
