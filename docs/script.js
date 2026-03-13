// Mobile hamburger menu toggle
const hamburger = document.querySelector('.hamburger');
const navMenu = document.querySelector('.nav-menu');

if (hamburger) {
  hamburger.addEventListener('click', () => {
    navMenu.classList.toggle('active');
  });

  // Close menu when clicking on a link
  document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      navMenu.classList.remove('active');
    });
  });
}

// Smooth scroll with offset for fixed navbar
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    e.preventDefault();
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      const offset = 80;
      const targetPosition = target.offsetTop - offset;
      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });
    }
  });
});

// Add active state to nav links on scroll
window.addEventListener('scroll', () => {
  let current = '';
  const sections = document.querySelectorAll('section[id]');

  sections.forEach(section => {
    const sectionTop = section.offsetTop;
    if (pageYOffset >= (sectionTop - 100)) {
      current = section.getAttribute('id');
    }
  });

  document.querySelectorAll('.nav-link').forEach(link => {
    link.classList.remove('active');
    if (link.getAttribute('href') === `#${current}`) {
      link.classList.add('active');
    }
  });
});

// Copy quickstart commands
document.querySelectorAll('.copy-code-btn').forEach(button => {
  button.addEventListener('click', async () => {
    const text = button.dataset.copy || '';
    if (!text) return;

    const originalLabel = button.textContent;
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = 'Copied';
      button.classList.add('copied');
    } catch (err) {
      button.textContent = 'Failed';
    }

    setTimeout(() => {
      button.textContent = originalLabel;
      button.classList.remove('copied');
    }, 1200);
  });
});

// Limitations accordion
document.querySelectorAll('.limitation-trigger').forEach(trigger => {
  trigger.addEventListener('click', () => {
    const row = trigger.closest('.limitation-row');
    const isOpen = row.classList.contains('open');
    row.classList.toggle('open', !isOpen);
    trigger.setAttribute('aria-expanded', String(!isOpen));
  });
});

// Capstone showcase poster toggle
const showcaseTrigger = document.querySelector('.showcase-trigger');

if (showcaseTrigger) {
  showcaseTrigger.addEventListener('click', () => {
    const panel = showcaseTrigger.closest('.showcase-panel');
    const isOpen = panel.classList.contains('open');
    panel.classList.toggle('open', !isOpen);
    showcaseTrigger.setAttribute('aria-expanded', String(!isOpen));
  });
}

// ─── Interactive Knowledge Graph ────────────────────────────────────────────

const kgShell = document.querySelector('[data-kg-shell]');

if (kgShell) {
  const stage       = kgShell.querySelector('[data-kg-stage]');
  const viewport    = kgShell.querySelector('[data-kg-viewport]');
  const edgesLayer  = kgShell.querySelector('[data-kg-edges]');
  const nodesLayer  = kgShell.querySelector('[data-kg-nodes]');
  const statsLabel  = kgShell.querySelector('[data-kg-stats]');
  const detailTitle = kgShell.querySelector('[data-kg-detail-title]');
  const detailType  = kgShell.querySelector('[data-kg-detail-type]');
  const detailBody  = kgShell.querySelector('[data-kg-detail-body]');
  const loadingEl   = kgShell.querySelector('[data-kg-loading]');
  const zoomInBtn   = kgShell.querySelector('[data-kg-zoom="in"]');
  const zoomOutBtn  = kgShell.querySelector('[data-kg-zoom="out"]');
  const resetBtn    = kgShell.querySelector('[data-kg-reset]');
  const edgeTooltip = document.getElementById('kg-edge-tooltip');

  const SVG_NS      = 'http://www.w3.org/2000/svg';
  const GRAPH_W     = 1600;
  const GRAPH_H     = 1000;

  let zoomLevel        = 1;
  let panX             = 0;
  let panY             = 0;
  let defaultZoom      = 1;
  let defaultPanX      = 0;
  let defaultPanY      = 0;
  let nodeElements     = [];
  let currentLayout    = null;
  let allTriples       = [];

  // Drag state
  let isDragging  = false;
  let dragStartX  = 0;
  let dragStartY  = 0;
  let dragOriginX = 0;
  let dragOriginY = 0;

  // Pinch state
  let pinchStartDist = 0;
  let pinchStartZoom = 1;

  // ── Transform helpers ─────────────────────────────────────────────────────

  const applyTransform = () => {
    viewport.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
  };

  const zoomAroundCenter = next => {
    if (!stage) return;
    const rect   = stage.getBoundingClientRect();
    const cx     = rect.width  / 2;
    const cy     = rect.height / 2;
    const worldX = (cx - panX) / zoomLevel;
    const worldY = (cy - panY) / zoomLevel;
    zoomLevel = next;
    panX = cx - worldX * zoomLevel;
    panY = cy - worldY * zoomLevel;
    applyTransform();
  };

  const zoomAroundPoint = (next, clientX, clientY) => {
    if (!stage) return;
    const rect   = stage.getBoundingClientRect();
    const px     = clientX - rect.left;
    const py     = clientY - rect.top;
    const worldX = (px - panX) / zoomLevel;
    const worldY = (py - panY) / zoomLevel;
    zoomLevel = next;
    panX = px - worldX * zoomLevel;
    panY = py - worldY * zoomLevel;
    applyTransform();
  };

  // ── Fit graph to stage ────────────────────────────────────────────────────

  const fitGraphToStage = layout => {
    const positions = [...layout.values()];
    if (!positions.length || !stage) return;
    currentLayout = layout;

    const padding   = 120;
    const minX      = Math.min(...positions.map(p => p.x)) - padding;
    const maxX      = Math.max(...positions.map(p => p.x)) + padding;
    const minY      = Math.min(...positions.map(p => p.y)) - padding;
    const maxY      = Math.max(...positions.map(p => p.y)) + padding;
    const gw        = maxX - minX;
    const gh        = maxY - minY;

    const sw        = stage.clientWidth  || 1;
    const sh        = stage.clientHeight || 1;
    defaultZoom     = Math.min(sw / gw, sh / gh, 1);
    defaultPanX     = (sw - gw * defaultZoom) / 2 - minX * defaultZoom;
    defaultPanY     = (sh - gh * defaultZoom) / 2 - minY * defaultZoom;

    zoomLevel = defaultZoom;
    panX      = defaultPanX;
    panY      = defaultPanY;
    applyTransform();
  };

  // ── Node helpers ──────────────────────────────────────────────────────────

  const truncateLabel = val => {
    if (!val) return '';
    return val.length <= 22 ? val : `${val.slice(0, 19).trim()}…`;
  };

  const typeToClass = type => {
    const n = (type || '').toLowerCase();
    if (n.includes('revenue') || n.includes('earnings') || n.includes('growth') ||
        n.includes('profit')  || n.includes('metric')   || n.includes('rate') ||
        n.includes('percentage')) return 'kg-node-metric';
    if (n.includes('product') || n.includes('application') || n.includes('subscription') ||
        n.includes('model')   || n.includes('solution')) return 'kg-node-product';
    if (n.includes('company')) return 'kg-node-core';
    return 'kg-node-theme';
  };

  // ── Detail panel ──────────────────────────────────────────────────────────

  const setActiveNode = node => {
    nodeElements.forEach(el => el.classList.toggle('selected', el === node));

    const name = node.dataset.nodeTitle || '';
    const type = node.dataset.nodeType  || '';
    detailTitle.textContent = name;
    detailType.textContent  = type;

    const related = allTriples.filter(t => t.head === name || t.tail === name);

    if (!related.length) {
      detailBody.innerHTML = `<span class="kg-triple-empty">${name} appears as a ${type || 'node'} in the extracted graph.</span>`;
      return;
    }

    const count = related.length;
    const rows = related.map(t => {
      const rel = t.relation.replace(/_/g, ' ');
      return `<li class="kg-triple-row">
        <span class="kg-triple-head">${t.head}</span>
        <span class="kg-triple-relation">${rel}</span>
        <span class="kg-triple-tail">${t.tail}</span>
      </li>`;
    }).join('');

    detailBody.innerHTML = `
      <p class="kg-triple-count"><strong>${count}</strong> extracted relationship${count === 1 ? '' : 's'}</p>
      <div class="kg-triple-legend" aria-label="Color key for triples">
        <span class="kg-triple-head">Subject</span>
        <span class="kg-triple-relation">predicate</span>
        <span class="kg-triple-tail">Object</span>
      </div>
      <div class="kg-triple-scroll"><ul class="kg-triple-list">${rows}</ul></div>`;
  };

  // ── Connected components ──────────────────────────────────────────────────

  const getConnectedComponents = adjacency => {
    const visited    = new Set();
    const components = [];

    adjacency.forEach((_, node) => {
      if (visited.has(node)) return;
      const stack     = [node];
      const component = [];
      visited.add(node);

      while (stack.length) {
        const current = stack.pop();
        component.push(current);
        (adjacency.get(current) || []).forEach(neighbor => {
          if (visited.has(neighbor)) return;
          visited.add(neighbor);
          stack.push(neighbor);
        });
      }
      components.push(component);
    });

    return components.sort((a, b) => b.length - a.length);
  };

  // ── Build graph ───────────────────────────────────────────────────────────

  const buildGraph = triples => {
    const headCounts = new Map();
    const nodeMap    = new Map();
    const adjacency  = new Map();

    triples.forEach(triple => {
      headCounts.set(triple.head, (headCounts.get(triple.head) || 0) + 1);
      if (!nodeMap.has(triple.head)) nodeMap.set(triple.head, { name: triple.head, type: triple.head_type || 'Entity', count: 0 });
      if (!nodeMap.has(triple.tail)) nodeMap.set(triple.tail, { name: triple.tail, type: triple.tail_type || 'Entity', count: 0 });
      if (!adjacency.has(triple.head)) adjacency.set(triple.head, new Set());
      if (!adjacency.has(triple.tail)) adjacency.set(triple.tail, new Set());
      adjacency.get(triple.head).add(triple.tail);
      adjacency.get(triple.tail).add(triple.head);
      nodeMap.get(triple.head).count += 1;
      nodeMap.get(triple.tail).count += 1;
    });

    const centralNode = [...headCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
    if (!centralNode) {
      detailTitle.textContent = 'No graph data';
      detailType.textContent  = 'Unavailable';
      detailBody.innerHTML    = '<span class="kg-triple-empty">The extraction result did not include any triples to visualize.</span>';
      loadingEl?.remove();
      return;
    }

    const components       = getConnectedComponents(adjacency);
    const mainIdx          = components.findIndex(c => c.includes(centralNode));
    const mainComponent    = mainIdx >= 0 ? components.splice(mainIdx, 1)[0] : [centralNode];
    const visibleTriples   = triples.filter(t => nodeMap.has(t.head) && nodeMap.has(t.tail));

    const layout  = new Map();
    const centerX = 760;
    const centerY = 500;
    edgesLayer.setAttribute('viewBox', `0 0 ${GRAPH_W} ${GRAPH_H}`);
    layout.set(centralNode, { x: centerX, y: centerY });

    const mainNeighbors = mainComponent
      .filter(n => n !== centralNode)
      .sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));

    const rings = [
      { radius: 240, capacity: 10 },
      { radius: 390, capacity: 14 },
      { radius: 540, capacity: 20 },
    ];

    let ringIdx = 0, idxInRing = 0;
    mainNeighbors.forEach(name => {
      while (ringIdx < rings.length - 1 && idxInRing >= rings[ringIdx].capacity) {
        ringIdx += 1; idxInRing = 0;
      }
      const ring  = rings[ringIdx];
      const angle = (-Math.PI / 2) + (idxInRing / Math.max(ring.capacity, 1)) * Math.PI * 2;
      layout.set(name, {
        x: centerX + Math.cos(angle) * ring.radius,
        y: centerY + Math.sin(angle) * ring.radius,
      });
      idxInRing += 1;
    });

    const componentAnchors = [
      { x: 230, y: 180 }, { x: 1320, y: 210 },
      { x: 250, y: 820 }, { x: 1320, y: 790 },
    ];

    components.forEach((comp, ci) => {
      const anchor = componentAnchors[ci % componentAnchors.length];
      const sorted = comp.slice().sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));
      layout.set(sorted[0], anchor);
      sorted.slice(1).forEach((name, li) => {
        const angle  = (-Math.PI / 2) + (li / Math.max(sorted.length - 1, 1)) * Math.PI * 2;
        layout.set(name, {
          x: anchor.x + Math.cos(angle) * 145,
          y: anchor.y + Math.sin(angle) * 145,
        });
      });
    });

    edgesLayer.innerHTML = '';
    nodesLayer.innerHTML = '';

    // SVG arrowhead marker
    const defs   = document.createElementNS(SVG_NS, 'defs');
    const marker = document.createElementNS(SVG_NS, 'marker');
    marker.setAttribute('id', 'kg-arrowhead');
    marker.setAttribute('markerUnits', 'userSpaceOnUse');
    marker.setAttribute('markerWidth',  '18');
    marker.setAttribute('markerHeight', '12');
    marker.setAttribute('refX', '18');
    marker.setAttribute('refY', '6');
    marker.setAttribute('orient', 'auto');
    const arrowPoly = document.createElementNS(SVG_NS, 'polygon');
    arrowPoly.setAttribute('points', '0 0, 18 6, 0 12');
    arrowPoly.setAttribute('fill', 'rgba(17, 24, 39, 0.35)');
    marker.appendChild(arrowPoly);
    defs.appendChild(marker);
    edgesLayer.appendChild(defs);

    // Helper: node radius in SVG coords
    const nodeRadius = name => {
      if (name === centralNode) return 64;
      return (nodeMap.get(name)?.count <= 1) ? 48 : 56;
    };

    // Aggregate all relation labels per edge pair, preserving first-seen direction
    const edgeRelations = new Map();
    visibleTriples.forEach(t => {
      if (!layout.get(t.head) || !layout.get(t.tail) || t.head === t.tail) return;
      const key = [t.head, t.tail].sort().join('::');
      if (!edgeRelations.has(key)) edgeRelations.set(key, { head: t.head, tail: t.tail, rels: [] });
      edgeRelations.get(key).rels.push(t.relation.replace(/_/g, ' '));
    });

    edgeRelations.forEach(({ head, tail, rels }) => {
      const start = layout.get(head);
      const end   = layout.get(tail);
      if (!start || !end) return;

      // Compute unit vector
      const dx  = end.x - start.x;
      const dy  = end.y - start.y;
      const len = Math.hypot(dx, dy);
      if (len < 1) return;
      const ux = dx / len;
      const uy = dy / len;

      // Offset from node center to node edge (arrowhead tip lands at target edge)
      const x1 = start.x + ux * nodeRadius(head);
      const y1 = start.y + uy * nodeRadius(head);
      const x2 = end.x   - ux * nodeRadius(tail);
      const y2 = end.y   - uy * nodeRadius(tail);

      const label = [...new Set(rels)].join(' · ');

      // Wide invisible hit-area for easy hover
      const hitLine = document.createElementNS(SVG_NS, 'line');
      hitLine.setAttribute('x1', x1); hitLine.setAttribute('y1', y1);
      hitLine.setAttribute('x2', x2); hitLine.setAttribute('y2', y2);
      hitLine.dataset.relation = label;
      hitLine.classList.add('kg-edge-hit');
      edgesLayer.appendChild(hitLine);

      // Visible directed line with arrowhead
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', x1); line.setAttribute('y1', y1);
      line.setAttribute('x2', x2); line.setAttribute('y2', y2);
      line.setAttribute('marker-end', 'url(#kg-arrowhead)');
      line.classList.add('kg-edge-visual');
      const titleEl = document.createElementNS(SVG_NS, 'title');
      titleEl.textContent = label;
      line.appendChild(titleEl);
      edgesLayer.appendChild(line);
    });

    const orderedNodes = [...nodeMap.values()].sort((a, b) => {
      if (a.name === centralNode) return -1;
      if (b.name === centralNode) return 1;
      return b.count - a.count;
    });

    nodeElements = orderedNodes.map(node => {
      const pos = layout.get(node.name);
      if (!pos) return null;

      const el  = document.createElement('button');
      el.type   = 'button';
      el.className = `kg-node ${typeToClass(node.type)}`;
      if (node.count <= 1 && node.name !== centralNode) el.classList.add('kg-node-minor');
      el.style.left = `${pos.x}px`;
      el.style.top  = `${pos.y}px`;
      el.dataset.nodeTitle = node.name;
      el.dataset.nodeType  = node.type;
      el.textContent = truncateLabel(node.name);
      el.addEventListener('click', () => setActiveNode(el));

      // Show full name tooltip on hover (only when label is truncated)
      if (node.name.length > 1 && edgeTooltip) {
        el.addEventListener('mouseenter', () => {
          edgeTooltip.textContent = node.name;
          edgeTooltip.classList.add('visible');
        });
        el.addEventListener('mousemove', e => {
          edgeTooltip.style.left = `${e.clientX + 14}px`;
          edgeTooltip.style.top  = `${e.clientY - 10}px`;
        });
        el.addEventListener('mouseleave', () => edgeTooltip.classList.remove('visible'));
      }

      nodesLayer.appendChild(el);
      return el;
    }).filter(Boolean);

    if (nodeElements[0]) setActiveNode(nodeElements[0]);

    loadingEl?.remove();
    fitGraphToStage(layout);
    requestAnimationFrame(() => fitGraphToStage(layout));
  };

  // ── Edge tooltip ──────────────────────────────────────────────────────────

  edgesLayer?.addEventListener('mousemove', e => {
    if (!edgeTooltip) return;
    // Walk up from target to find an element with data-relation (hit-area line)
    let el = e.target;
    let relation = null;
    while (el && el !== edgesLayer) {
      if (el.dataset && el.dataset.relation) { relation = el.dataset.relation; break; }
      el = el.parentElement;
    }
    if (!relation) {
      edgeTooltip.classList.remove('visible');
      return;
    }
    edgeTooltip.textContent = relation;
    edgeTooltip.classList.add('visible');
    edgeTooltip.style.left = `${e.clientX + 14}px`;
    edgeTooltip.style.top  = `${e.clientY - 10}px`;
  });

  edgesLayer?.addEventListener('mouseleave', () => edgeTooltip?.classList.remove('visible'));

  // ── Zoom controls ─────────────────────────────────────────────────────────

  zoomInBtn?.addEventListener('click', () =>
    zoomAroundCenter(Math.min(2.8, +(zoomLevel + 0.12).toFixed(2))));

  zoomOutBtn?.addEventListener('click', () =>
    zoomAroundCenter(Math.max(defaultZoom, +(zoomLevel - 0.12).toFixed(2))));

  resetBtn?.addEventListener('click', () => {
    zoomLevel = defaultZoom;
    panX      = defaultPanX;
    panY      = defaultPanY;
    applyTransform();
  });

  // Scroll-to-zoom (slowed down: 0.06 per notch)
  stage?.addEventListener('wheel', e => {
    e.preventDefault();
    const dir  = e.deltaY < 0 ? 1 : -1;
    const next = dir > 0
      ? Math.min(2.8, +(zoomLevel + 0.06).toFixed(2))
      : Math.max(defaultZoom, +(zoomLevel - 0.06).toFixed(2));
    zoomAroundPoint(next, e.clientX, e.clientY);
  }, { passive: false });

  // ── Drag to pan ───────────────────────────────────────────────────────────

  stage?.addEventListener('pointerdown', e => {
    if (e.button !== 0 || e.target.closest('.kg-node')) return;
    isDragging  = true;
    dragStartX  = e.clientX;
    dragStartY  = e.clientY;
    dragOriginX = panX;
    dragOriginY = panY;
    stage.classList.add('dragging');
  });

  window.addEventListener('pointermove', e => {
    if (!isDragging) return;
    panX = dragOriginX + (e.clientX - dragStartX);
    panY = dragOriginY + (e.clientY - dragStartY);
    applyTransform();
  });

  window.addEventListener('pointerup', () => {
    if (!isDragging) return;
    isDragging = false;
    stage?.classList.remove('dragging');
  });

  // ── Pinch-to-zoom (mobile) ────────────────────────────────────────────────

  stage?.addEventListener('touchstart', e => {
    if (e.touches.length !== 2) return;
    const dx = e.touches[0].clientX - e.touches[1].clientX;
    const dy = e.touches[0].clientY - e.touches[1].clientY;
    pinchStartDist = Math.hypot(dx, dy);
    pinchStartZoom = zoomLevel;
  }, { passive: true });

  stage?.addEventListener('touchmove', e => {
    if (e.touches.length !== 2) return;
    e.preventDefault();
    const dx   = e.touches[0].clientX - e.touches[1].clientX;
    const dy   = e.touches[0].clientY - e.touches[1].clientY;
    const dist = Math.hypot(dx, dy);
    const midX = (e.touches[0].clientX + e.touches[1].clientX) / 2;
    const midY = (e.touches[0].clientY + e.touches[1].clientY) / 2;
    const next = Math.min(2.8, Math.max(defaultZoom, pinchStartZoom * (dist / pinchStartDist)));
    zoomAroundPoint(next, midX, midY);
  }, { passive: false });

  // ── Responsive: re-fit on container resize ────────────────────────────────

  if (typeof ResizeObserver !== 'undefined' && stage) {
    new ResizeObserver(() => {
      if (currentLayout) fitGraphToStage(currentLayout);
    }).observe(stage);
  } else {
    window.addEventListener('resize', () => {
      if (currentLayout) fitGraphToStage(currentLayout);
    });
  }

  applyTransform();

  // ── Fetch graph data ──────────────────────────────────────────────────────

  fetch('interactive/extraction_result.json')
    .then(res => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    })
    .then(data => {
      allTriples = data.flatMap(item => item.triple_list || []);
      const nodeNames = new Set();
      allTriples.forEach(t => { nodeNames.add(t.head); nodeNames.add(t.tail); });
      if (statsLabel) statsLabel.textContent = `${nodeNames.size} nodes · ${allTriples.length} relations`;
      buildGraph(allTriples);
    })
    .catch(() => {
      detailTitle.textContent = 'Graph unavailable';
      detailType.textContent  = 'Load error';
      detailBody.innerHTML    = '<span class="kg-triple-empty">The interactive graph could not be loaded.</span>';
      loadingEl?.remove();
    });
}
