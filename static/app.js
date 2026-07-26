// NexusAI Assistant Client Engine
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initFileUpload();
    loadDocuments();
    initChat();
    initAnalysis();
    initAnalytics();
    initMLSandbox();
});

let currentSessionId = null;
let currentRetrievedContext = [];

// 1. Navigation Tab Management
function initNavigation() {
    const navButtons = document.querySelectorAll('.nav-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const headerTitle = document.getElementById('header-title');
    const headerSubtitle = document.getElementById('header-subtitle');

    const tabMeta = {
        'documents-tab': { title: 'Document Workspace', subtitle: 'Upload, process, and auto-classify technical research papers' },
        'chat-tab': { title: 'RAG Research Assistant', subtitle: 'Ask grounded questions with strict page-level citations' },
        'analysis-tab': { title: 'Multi-Document Intelligence', subtitle: 'Compare research methodologies, pros/cons, and multi-tier summaries' },
        'analytics-tab': { title: 'System Analytics & Metrics', subtitle: 'Track index volume, query latency, and TensorFlow category distribution' },
        'ml-tab': { title: 'TensorFlow Classifier Sandbox', subtitle: 'Test neural network domain predictions on raw text passages' }
    };

    navButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');

            navButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            if (tabMeta[targetTab]) {
                headerTitle.textContent = tabMeta[targetTab].title;
                headerSubtitle.textContent = tabMeta[targetTab].subtitle;
            }

            if (targetTab === 'analytics-tab') loadAnalytics();
            if (targetTab === 'documents-tab') loadDocuments();
            if (targetTab === 'analysis-tab') loadAnalysisDocPicker();
        });
    });

    document.getElementById('refresh-btn').addEventListener('click', () => {
        loadDocuments();
        loadAnalytics();
    });
}

// 2. File Upload & Document Management
function initFileUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadSpinner = document.getElementById('upload-spinner');

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = '#6366F1';
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'rgba(99, 102, 241, 0.4)';
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'rgba(99, 102, 241, 0.4)';
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });
}

async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        alert('Only PDF files are supported.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    const uploadSpinner = document.getElementById('upload-spinner');
    uploadSpinner.classList.remove('hidden');

    try {
        const response = await fetch('/documents/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (response.ok) {
            loadDocuments();
            // Poll document status for 10 seconds
            let attempts = 0;
            const pollInterval = setInterval(async () => {
                attempts++;
                await loadDocuments();
                if (attempts >= 5) clearInterval(pollInterval);
            }, 2000);
        } else {
            alert('Upload Error: ' + (data.detail || 'Upload failed.'));
        }
    } catch (err) {
        console.error('Upload Error:', err);
        alert('Upload failed. Check server connection.');
    } finally {
        uploadSpinner.classList.add('hidden');
    }
}

async function loadDocuments() {
    try {
        const res = await fetch('/documents');
        const docs = await res.json();

        const tbody = document.getElementById('documents-table-body');
        const badge = document.getElementById('doc-count-badge');
        badge.textContent = `${docs.length} Documents`;

        if (!docs || docs.length === 0) {
            tbody.innerHTML = `<tr><td colspan="7" class="text-center py-4" style="text-align:center;">No documents uploaded yet. Upload a PDF to begin.</td></tr>`;
            return;
        }

        tbody.innerHTML = docs.map(d => {
            let statusBadge = `<span class="badge badge-warning">${d.processing_status}</span>`;
            if (d.processing_status === 'PROCESSED') statusBadge = `<span class="badge badge-success">Processed</span>`;
            if (d.processing_status === 'FAILED') statusBadge = `<span class="badge badge-danger">Failed</span>`;

            const categoryBadge = `<span class="badge badge-category"><i class="fa-solid fa-tag"></i> ${d.category}</span>`;
            const dateStr = d.upload_timestamp ? new Date(d.upload_timestamp).toLocaleDateString() : 'N/A';

            return `
                <tr>
                    <td><strong><i class="fa-solid fa-file-pdf" style="color:#EF4444; margin-right:6px;"></i> ${d.file_name}</strong></td>
                    <td>${statusBadge}</td>
                    <td>${categoryBadge}</td>
                    <td>${d.total_pages || 0}</td>
                    <td>${d.total_chunks || 0}</td>
                    <td>${dateStr}</td>
                    <td>
                        <button class="btn btn-outline" onclick="deleteDocument('${d.doc_id}')" title="Delete"><i class="fa-solid fa-trash"></i></button>
                        <button class="btn btn-outline" onclick="reprocessDocument('${d.doc_id}')" title="Reprocess"><i class="fa-solid fa-rotate-right"></i></button>
                    </td>
                </tr>
            `;
        }).join('');

        populateChatDocFilter(docs);
    } catch (err) {
        console.error('Failed to load documents:', err);
    }
}

async function deleteDocument(docId) {
    if (!confirm('Are you sure you want to delete this document and its vector embeddings?')) return;
    try {
        const res = await fetch(`/documents/${docId}`, { method: 'DELETE' });
        if (res.ok) {
            loadDocuments();
            loadAnalytics();
        }
    } catch (err) {
        console.error('Delete error:', err);
    }
}

async function reprocessDocument(docId) {
    try {
        const res = await fetch(`/documents/${docId}/reprocess`, { method: 'POST' });
        if (res.ok) {
            loadDocuments();
        }
    } catch (err) {
        console.error('Reprocess error:', err);
    }
}

// 3. RAG Chat & Session Management
function initChat() {
    loadSessions();

    document.getElementById('new-session-btn').addEventListener('click', async () => {
        await createNewSession();
    });

    const chatForm = document.getElementById('chat-form');
    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const input = document.getElementById('chat-input');
        const query = input.value.trim();
        if (!query) return;

        input.value = '';
        appendMessage('user', query);

        // Get selected doc filter
        const select = document.getElementById('chat-doc-filter');
        const selectedDocs = Array.from(select.selectedOptions).map(opt => opt.value);

        // Append loading message
        const loadingId = appendLoadingMessage();

        try {
            const res = await fetch('/search/qa', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    session_id: currentSessionId,
                    doc_ids: selectedDocs.length > 0 ? selectedDocs : null
                })
            });

            const data = await res.json();
            removeMessage(loadingId);

            if (res.ok) {
                currentSessionId = data.session_id;
                currentRetrievedContext = data.retrieved_context || [];
                appendMessage('assistant', data.answer, data.citations, data.retrieved_context);
                loadSessions();
            } else {
                appendMessage('assistant', 'Error: ' + (data.detail || 'Could not process question.'));
            }
        } catch (err) {
            removeMessage(loadingId);
            appendMessage('assistant', 'Network Error: Unable to complete QA query.');
        }
    });
}

async function createNewSession() {
    try {
        const res = await fetch('/search/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'Research Session' })
        });
        const data = await res.json();
        currentSessionId = data.session_id;
        document.getElementById('chat-messages').innerHTML = `
            <div class="message assistant-msg">
                <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="msg-body">
                    <p>New session initialized. How can I assist your research today?</p>
                </div>
            </div>
        `;
        loadSessions();
    } catch (err) {
        console.error('Session error:', err);
    }
}

async function loadSessions() {
    try {
        const res = await fetch('/search/sessions');
        const sessions = await res.json();
        const list = document.getElementById('session-list');

        if (!sessions || sessions.length === 0) {
            list.innerHTML = `<div style="font-size:12px; color:var(--text-muted); text-align:center; padding:10px;">No sessions yet.</div>`;
            return;
        }

        if (!currentSessionId && sessions.length > 0) {
            currentSessionId = sessions[0].session_id;
        }

        list.innerHTML = sessions.map(s => `
            <div class="session-item ${s.session_id === currentSessionId ? 'active' : ''}" onclick="switchSession('${s.session_id}')">
                <i class="fa-regular fa-message" style="margin-right:6px;"></i> ${s.title || 'Research Session'}
            </div>
        `).join('');
    } catch (err) {
        console.error('Failed to load sessions:', err);
    }
}

async function switchSession(sessionId) {
    currentSessionId = sessionId;
    loadSessions();
    try {
        const res = await fetch(`/search/sessions/${sessionId}/messages`);
        const messages = await res.json();
        const box = document.getElementById('chat-messages');
        box.innerHTML = '';

        if (!messages || messages.length === 0) {
            box.innerHTML = `
                <div class="message assistant-msg">
                    <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
                    <div class="msg-body"><p>Session loaded. Ask any question grounded in your repository.</p></div>
                </div>
            `;
            return;
        }

        messages.forEach(m => {
            appendMessage(m.role, m.content, m.citations || []);
        });
    } catch (err) {
        console.error('Failed to switch session:', err);
    }
}

function appendMessage(role, content, citations = [], context = []) {
    const box = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role === 'user' ? 'user-msg' : 'assistant-msg'}`;

    const avatar = role === 'user' ? `<i class="fa-solid fa-user"></i>` : `<i class="fa-solid fa-robot"></i>`;
    
    let citationsHtml = '';
    if (citations && citations.length > 0) {
        citationsHtml = `<div class="citation-pills">` + 
            citations.map(c => `
                <span class="citation-pill" onclick="openContextModal('${c.document}', ${c.page})">
                    <i class="fa-solid fa-bookmark"></i> ${c.document} (P. ${c.page})
                </span>
            `).join('') + `</div>`;
    }

    const formattedContent = content.replace(/\n/g, '<br>');

    msgDiv.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-body">
            <div>${formattedContent}</div>
            ${citationsHtml}
        </div>
    `;

    box.appendChild(msgDiv);
    box.scrollTop = box.scrollHeight;
}

function appendLoadingMessage() {
    const box = document.getElementById('chat-messages');
    const id = 'msg-loading-' + Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message assistant-msg';
    msgDiv.id = id;
    msgDiv.innerHTML = `
        <div class="msg-avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="msg-body">
            <div style="display:flex; align-items:center; gap:8px;">
                <div class="spinner"></div> Grounding RAG response across indexed vectors...
            </div>
        </div>
    `;
    box.appendChild(msgDiv);
    box.scrollTop = box.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function populateChatDocFilter(docs) {
    const select = document.getElementById('chat-doc-filter');
    select.innerHTML = docs.map(d => `
        <option value="${d.doc_id}">${d.file_name}</option>
    `).join('');
}

// Context Modal
function openContextModal(docName, pageNo) {
    const modal = document.getElementById('context-modal');
    const body = document.getElementById('modal-context-body');
    
    let filteredContext = currentRetrievedContext.filter(c => c.file_name === docName && c.page_number === pageNo);
    if (filteredContext.length === 0) filteredContext = currentRetrievedContext;

    body.innerHTML = filteredContext.map(c => `
        <div class="chunk-card">
            <strong><i class="fa-solid fa-file-pdf"></i> ${c.file_name} (Page ${c.page_number})</strong>
            <p style="margin-top:6px; color:var(--text-secondary); line-height:1.5;">${c.text}</p>
        </div>
    `).join('') || '<p>No detailed chunk context cached for this turn.</p>';

    modal.classList.remove('hidden');
}

function closeContextModal() {
    document.getElementById('context-modal').classList.add('hidden');
}

// 4. Multi-Document Analysis & Summarization
function initAnalysis() {
    loadAnalysisDocPicker();

    document.getElementById('btn-run-summary').addEventListener('click', async () => {
        const selected = getSelectedAnalysisDocs();
        if (selected.length === 0) {
            alert('Please select at least 1 document to summarize.');
            return;
        }

        const resultsContainer = document.getElementById('analysis-results');
        resultsContainer.innerHTML = `<div class="placeholder-state"><div class="spinner" style="margin:0 auto 10px;"></div><p>Generating structured multi-tier summary...</p></div>`;

        try {
            const res = await fetch('/analysis/summarize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ doc_id: selected[0] })
            });

            const data = await res.json();
            renderSummaryResult(data);
        } catch (err) {
            console.error('Summary error:', err);
            resultsContainer.innerHTML = `<p class="text-danger">Failed to generate document summary.</p>`;
        }
    });

    document.getElementById('btn-run-comparison').addEventListener('click', async () => {
        const selected = getSelectedAnalysisDocs();
        if (selected.length < 2) {
            alert('Please select at least 2 documents to run comparative analysis.');
            return;
        }

        const resultsContainer = document.getElementById('analysis-results');
        resultsContainer.innerHTML = `<div class="placeholder-state"><div class="spinner" style="margin:0 auto 10px;"></div><p>Generating side-by-side comparative matrix...</p></div>`;

        try {
            const res = await fetch('/analysis/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ doc_ids: selected })
            });

            const data = await res.json();
            renderComparisonResult(data);
        } catch (err) {
            console.error('Comparison error:', err);
            resultsContainer.innerHTML = `<p class="text-danger">Failed to compare selected documents.</p>`;
        }
    });
}

async function loadAnalysisDocPicker() {
    try {
        const res = await fetch('/documents');
        const docs = await res.json();
        const picker = document.getElementById('analysis-doc-picker');

        if (!docs || docs.length === 0) {
            picker.innerHTML = `<p style="font-size:13px; color:var(--text-muted);">No uploaded documents available.</p>`;
            return;
        }

        picker.innerHTML = docs.map(d => `
            <label class="doc-pick-item">
                <input type="checkbox" value="${d.doc_id}" class="analysis-doc-checkbox">
                <span><strong>${d.file_name}</strong> (${d.category})</span>
            </label>
        `).join('');
    } catch (err) {
        console.error('Analysis picker error:', err);
    }
}

function getSelectedAnalysisDocs() {
    const checkboxes = document.querySelectorAll('.analysis-doc-checkbox:checked');
    return Array.from(checkboxes).map(cb => cb.value);
}

function renderSummaryResult(data) {
    const container = document.getElementById('analysis-results');
    
    const takeawaysList = (data.key_takeaways || []).map(t => `<li><i class="fa-solid fa-check" style="color:var(--accent-emerald);"></i> ${t}</li>`).join('');
    const topicList = (data.topic_breakdown || []).map(t => `<li><i class="fa-solid fa-angle-right" style="color:var(--primary-indigo);"></i> ${t}</li>`).join('');

    container.innerHTML = `
        <div class="card mb-4">
            <div class="card-header">
                <h3><i class="fa-solid fa-file-contract"></i> Multi-Tier Summary: ${data.file_name}</h3>
            </div>
            
            <div style="margin-bottom:20px;">
                <h4 style="color:var(--primary-indigo); margin-bottom:6px;">Executive Summary</h4>
                <p style="color:var(--text-secondary); line-height:1.6;">${data.executive_summary}</p>
            </div>

            <div style="margin-bottom:20px;">
                <h4 style="color:var(--primary-purple); margin-bottom:6px;">Technical Breakdown</h4>
                <p style="color:var(--text-secondary); line-height:1.6;">${data.technical_summary}</p>
            </div>

            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;">
                <div>
                    <h4 style="color:var(--accent-emerald); margin-bottom:8px;">Key Takeaways</h4>
                    <ul style="list-style:none; display:flex; flex-direction:column; gap:6px;">${takeawaysList}</ul>
                </div>
                <div>
                    <h4 style="color:var(--accent-amber); margin-bottom:8px;">Structural Topics</h4>
                    <ul style="list-style:none; display:flex; flex-direction:column; gap:6px;">${topicList}</ul>
                </div>
            </div>
        </div>
    `;
}

function renderComparisonResult(data) {
    const container = document.getElementById('analysis-results');
    const comp = data.comparison || {};

    const methodRows = Object.entries(comp.methodologies || {}).map(([fn, val]) => `
        <tr><td><strong>${fn}</strong></td><td>${val}</td></tr>
    `).join('');

    const advRows = Object.entries(comp.advantages || {}).map(([fn, val]) => `
        <tr><td><strong>${fn}</strong></td><td>${val}</td></tr>
    `).join('');

    const similarities = (comp.similarities || []).map(s => `<li>• ${s}</li>`).join('');
    const differences = (comp.key_differences || []).map(d => `<li>• ${d}</li>`).join('');

    container.innerHTML = `
        <div class="card mb-4">
            <div class="card-header">
                <h3><i class="fa-solid fa-code-compare"></i> Side-by-Side Comparative Matrix (${data.document_count} Documents)</h3>
            </div>

            <h4 class="mb-2" style="color:var(--primary-indigo);">Methodologies & Approaches</h4>
            <table class="data-table mb-4">
                <thead><tr><th>Document</th><th>Methodology Details</th></tr></thead>
                <tbody>${methodRows}</tbody>
            </table>

            <h4 class="mb-2" style="color:var(--accent-emerald);">Advantages & Coverage</h4>
            <table class="data-table mb-4">
                <thead><tr><th>Document</th><th>Key Strengths</th></tr></thead>
                <tbody>${advRows}</tbody>
            </table>

            <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px;" class="mt-4">
                <div class="card" style="background:var(--bg-dark);">
                    <h4 style="color:var(--primary-purple); margin-bottom:8px;">Common Similarities</h4>
                    <ul style="list-style:none; display:flex; flex-direction:column; gap:6px;">${similarities}</ul>
                </div>
                <div class="card" style="background:var(--bg-dark);">
                    <h4 style="color:var(--accent-pink); margin-bottom:8px;">Key Differences</h4>
                    <ul style="list-style:none; display:flex; flex-direction:column; gap:6px;">${differences}</ul>
                </div>
            </div>
        </div>
    `;
}

// 5. System Analytics
function initAnalytics() {
    loadAnalytics();
}

async function loadAnalytics() {
    try {
        const statsRes = await fetch('/analytics/stats');
        const stats = await statsRes.json();

        document.getElementById('stat-total-docs').textContent = stats.total_documents || 0;
        document.getElementById('stat-total-chunks').textContent = stats.total_indexed_chunks || 0;
        document.getElementById('stat-total-queries').textContent = stats.total_queries_executed || 0;
        document.getElementById('stat-avg-latency').textContent = `${stats.average_latency_ms || 0} ms`;

        // Render TF Category Distribution
        const catBox = document.getElementById('category-distribution-box');
        const dist = stats.category_distribution || {};
        const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

        catBox.innerHTML = Object.entries(dist).map(([cat, count]) => {
            const pct = Math.round((count / total) * 100);
            return `
                <div class="prob-bar-item">
                    <div class="prob-bar-label"><span>${cat}</span><span>${count} docs (${pct}%)</span></div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                    </div>
                </div>
            `;
        }).join('') || '<p style="font-size:13px; color:var(--text-muted);">No document categories recorded yet.</p>';

        // Render Top Documents
        const topRes = await fetch('/analytics/top-documents');
        const topDocs = await topRes.json();
        const topBody = document.getElementById('top-docs-table-body');

        topBody.innerHTML = topDocs.map(d => `
            <tr>
                <td><strong>${d.file_name}</strong></td>
                <td><span class="badge badge-category">${d.category}</span></td>
                <td>${d.total_chunks}</td>
                <td>${d.total_pages}</td>
            </tr>
        `).join('') || '<tr><td colspan="4">No documents indexed yet.</td></tr>';

    } catch (err) {
        console.error('Analytics load error:', err);
    }
}

// 6. TensorFlow Classifier Sandbox
function initMLSandbox() {
    document.getElementById('btn-predict-ml').addEventListener('click', async () => {
        const input = document.getElementById('ml-input-text').value.trim();
        if (!input) {
            alert('Please enter a text passage to classify.');
            return;
        }

        try {
            const res = await fetch('/analysis/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input })
            });

            const data = await res.json();
            if (res.ok) {
                document.getElementById('ml-prediction-results').classList.remove('hidden');
                document.getElementById('pred-category-name').textContent = data.category;
                document.getElementById('pred-confidence').textContent = `Confidence: ${Math.round(data.confidence * 100)}%`;

                const barsContainer = document.getElementById('prob-bars-container');
                const probs = data.probabilities || {};

                barsContainer.innerHTML = Object.entries(probs).map(([cat, prob]) => {
                    const pct = Math.round(prob * 100);
                    return `
                        <div class="prob-bar-item">
                            <div class="prob-bar-label">
                                <span>${cat}</span>
                                <span>${pct}%</span>
                            </div>
                            <div class="prob-bar-track">
                                <div class="prob-bar-fill" style="width: ${pct}%;"></div>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        } catch (err) {
            console.error('ML Sandbox error:', err);
        }
    });
}
