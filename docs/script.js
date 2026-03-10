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
    const sectionHeight = section.clientHeight;
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

// Interactive knowledge graph preview
const kgShell = document.querySelector('[data-kg-shell]');

if (kgShell) {
  const stage = kgShell.querySelector('[data-kg-stage]');
  const viewport = kgShell.querySelector('[data-kg-viewport]');
  const edgesLayer = kgShell.querySelector('[data-kg-edges]');
  const nodesLayer = kgShell.querySelector('[data-kg-nodes]');
  const statsLabel = kgShell.querySelector('[data-kg-stats]');
  const detailTitle = kgShell.querySelector('[data-kg-detail-title]');
  const detailType = kgShell.querySelector('[data-kg-detail-type]');
  const detailBody = kgShell.querySelector('[data-kg-detail-body]');
  const zoomInButton = kgShell.querySelector('[data-kg-zoom="in"]');
  const zoomOutButton = kgShell.querySelector('[data-kg-zoom="out"]');
  const resetButton = kgShell.querySelector('[data-kg-reset]');
  const SVG_NS = 'http://www.w3.org/2000/svg';
  const GRAPH_WIDTH = 1600;
  const GRAPH_HEIGHT = 1000;

  let zoomLevel = 1;
  let panX = 0;
  let panY = 0;
  let defaultZoomLevel = 1;
  let defaultPanX = 0;
  let defaultPanY = 0;
  let nodeElements = [];
  let currentLayout = null;
  let isDragging = false;
  let dragStartX = 0;
  let dragStartY = 0;
  let dragOriginX = 0;
  let dragOriginY = 0;

  const applyTransform = () => {
    viewport.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
  };

  const setActiveNode = node => {
    nodeElements.forEach(item => item.classList.toggle('selected', item === node));
    detailTitle.textContent = node.dataset.nodeTitle || '';
    detailType.textContent = node.dataset.nodeType || '';
    detailBody.textContent = node.dataset.nodeDetail || '';
  };

  const truncateLabel = value => {
    if (!value) return '';
    if (value.length <= 22) return value;
    return `${value.slice(0, 19).trim()}...`;
  };

  const typeToClass = type => {
    const normalized = (type || '').toLowerCase();
    if (
      normalized.includes('revenue') ||
      normalized.includes('earnings') ||
      normalized.includes('growth') ||
      normalized.includes('profit') ||
      normalized.includes('metric') ||
      normalized.includes('rate') ||
      normalized.includes('percentage')
    ) {
      return 'kg-node-metric';
    }
    if (
      normalized.includes('product') ||
      normalized.includes('application') ||
      normalized.includes('subscription') ||
      normalized.includes('model') ||
      normalized.includes('solution')
    ) {
      return 'kg-node-product';
    }
    if (normalized.includes('company')) {
      return 'kg-node-core';
    }
    return 'kg-node-theme';
  };

  const summarizeNode = (nodeName, nodeType, triples) => {
    const related = triples.filter(triple => triple.head === nodeName || triple.tail === nodeName);
    if (!related.length) {
      return `${nodeName} appears in the extracted graph as a ${nodeType || 'node'}.`;
    }

    const summary = related
      .slice(0, 3)
      .map(triple => {
        if (triple.head === nodeName) {
          return `${triple.relation.replace(/_/g, ' ')} ${triple.tail}`;
        }
        return `${triple.head} ${triple.relation.replace(/_/g, ' ')} this node`;
      })
      .join('; ');

    return `${related.length} extracted relationship${related.length === 1 ? '' : 's'}: ${summary}.`;
  };

  const getConnectedComponents = adjacency => {
    const visited = new Set();
    const components = [];

    adjacency.forEach((_, node) => {
      if (visited.has(node)) return;
      const stack = [node];
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

  const fitGraphToStage = layout => {
    const positions = [...layout.values()];
    if (!positions.length || !stage) return;
    currentLayout = layout;

    const padding = 120;
    const minX = Math.min(...positions.map(point => point.x)) - padding;
    const maxX = Math.max(...positions.map(point => point.x)) + padding;
    const minY = Math.min(...positions.map(point => point.y)) - padding;
    const maxY = Math.max(...positions.map(point => point.y)) + padding;
    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;

    const stageWidth = stage.clientWidth || 1;
    const stageHeight = stage.clientHeight || 1;
    defaultZoomLevel = Math.min(stageWidth / graphWidth, stageHeight / graphHeight, 1);
    defaultPanX = (stageWidth - graphWidth * defaultZoomLevel) / 2 - minX * defaultZoomLevel;
    defaultPanY = (stageHeight - graphHeight * defaultZoomLevel) / 2 - minY * defaultZoomLevel;

    zoomLevel = defaultZoomLevel;
    panX = defaultPanX;
    panY = defaultPanY;
    applyTransform();
  };

  const zoomAroundCenter = nextZoomLevel => {
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    const worldX = (centerX - panX) / zoomLevel;
    const worldY = (centerY - panY) / zoomLevel;

    zoomLevel = nextZoomLevel;
    panX = centerX - worldX * zoomLevel;
    panY = centerY - worldY * zoomLevel;
    applyTransform();
  };

  const zoomAroundPoint = (nextZoomLevel, clientX, clientY) => {
    if (!stage) return;
    const rect = stage.getBoundingClientRect();
    const pointX = clientX - rect.left;
    const pointY = clientY - rect.top;
    const worldX = (pointX - panX) / zoomLevel;
    const worldY = (pointY - panY) / zoomLevel;

    zoomLevel = nextZoomLevel;
    panX = pointX - worldX * zoomLevel;
    panY = pointY - worldY * zoomLevel;
    applyTransform();
  };

  const buildGraph = triples => {
    const headCounts = new Map();
    const nodeMap = new Map();
    const adjacency = new Map();

    triples.forEach(triple => {
      headCounts.set(triple.head, (headCounts.get(triple.head) || 0) + 1);
      if (!nodeMap.has(triple.head)) {
        nodeMap.set(triple.head, { name: triple.head, type: triple.head_type || 'Entity', count: 0 });
      }
      if (!nodeMap.has(triple.tail)) {
        nodeMap.set(triple.tail, { name: triple.tail, type: triple.tail_type || 'Entity', count: 0 });
      }
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
      detailType.textContent = 'Unavailable';
      detailBody.textContent = 'The extraction result did not include any triples to visualize.';
      return;
    }

    const components = getConnectedComponents(adjacency);
    const mainComponentIndex = components.findIndex(component => component.includes(centralNode));
    const mainComponent = mainComponentIndex >= 0 ? components.splice(mainComponentIndex, 1)[0] : [centralNode];
    const visibleTriples = triples.filter(triple => nodeMap.has(triple.head) && nodeMap.has(triple.tail));

    const layout = new Map();
    edgesLayer.setAttribute('viewBox', `0 0 ${GRAPH_WIDTH} ${GRAPH_HEIGHT}`);
    const centerX = 760;
    const centerY = 500;
    layout.set(centralNode, { x: centerX, y: centerY });

    const mainNeighbors = mainComponent
      .filter(name => name !== centralNode)
      .sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));

    const rings = [
      { radius: 240, capacity: 10 },
      { radius: 390, capacity: 14 },
      { radius: 540, capacity: 20 },
    ];

    let ringIndex = 0;
    let indexInRing = 0;
    mainNeighbors.forEach(nodeName => {
      while (ringIndex < rings.length - 1 && indexInRing >= rings[ringIndex].capacity) {
        ringIndex += 1;
        indexInRing = 0;
      }

      const ring = rings[ringIndex];
      const angle = (-Math.PI / 2) + (indexInRing / Math.max(ring.capacity, 1)) * Math.PI * 2;
      layout.set(nodeName, {
        x: centerX + Math.cos(angle) * ring.radius,
        y: centerY + Math.sin(angle) * ring.radius,
      });
      indexInRing += 1;
    });

    const componentAnchors = [
      { x: 230, y: 180 },
      { x: 1320, y: 210 },
      { x: 250, y: 820 },
      { x: 1320, y: 790 },
    ];

    components.forEach((component, componentIndex) => {
      const anchor = componentAnchors[componentIndex % componentAnchors.length];
      const sortedComponent = component
        .slice()
        .sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));

      const componentCenter = sortedComponent[0];
      layout.set(componentCenter, anchor);

      const leaves = sortedComponent.slice(1);
      leaves.forEach((nodeName, leafIndex) => {
        const angle = (-Math.PI / 2) + (leafIndex / Math.max(leaves.length, 1)) * Math.PI * 2;
        const radius = 145;
        layout.set(nodeName, {
          x: anchor.x + Math.cos(angle) * radius,
          y: anchor.y + Math.sin(angle) * radius,
        });
      });
    });

    edgesLayer.innerHTML = '';
    nodesLayer.innerHTML = '';

    const edgeKeys = new Set();
    visibleTriples.forEach(triple => {
      const start = layout.get(triple.head);
      const end = layout.get(triple.tail);
      if (!start || !end || (triple.head === triple.tail)) return;
      const edgeKey = [triple.head, triple.tail].sort().join('::');
      if (edgeKeys.has(edgeKey)) return;
      edgeKeys.add(edgeKey);

      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', start.x);
      line.setAttribute('y1', start.y);
      line.setAttribute('x2', end.x);
      line.setAttribute('y2', end.y);
      edgesLayer.appendChild(line);
    });

    const orderedNodes = [...nodeMap.values()].sort((a, b) => {
      if (a.name === centralNode) return -1;
      if (b.name === centralNode) return 1;
      return b.count - a.count;
    });

    nodeElements = orderedNodes.map(node => {
      const position = layout.get(node.name);
      if (!position) return null;
      const element = document.createElement('button');
      element.type = 'button';
      element.className = `kg-node ${typeToClass(node.type)}`;
      if (node.count <= 1 && node.name !== centralNode) {
        element.classList.add('kg-node-minor');
      }
      element.style.left = `${position.x}px`;
      element.style.top = `${position.y}px`;
      element.dataset.nodeTitle = node.name;
      element.dataset.nodeType = node.type;
      element.dataset.nodeDetail = summarizeNode(node.name, node.type, triples);
      element.textContent = truncateLabel(node.name);
      element.addEventListener('click', () => setActiveNode(element));
      nodesLayer.appendChild(element);
      return element;
    }).filter(Boolean);

    if (nodeElements[0]) {
      nodeElements[0].classList.add('selected');
      detailTitle.textContent = nodeElements[0].dataset.nodeTitle || '';
      detailType.textContent = nodeElements[0].dataset.nodeType || '';
      detailBody.textContent = nodeElements[0].dataset.nodeDetail || '';
    }

    fitGraphToStage(layout);
  };

  zoomInButton?.addEventListener('click', () => {
    zoomAroundCenter(Math.min(2.8, +(zoomLevel + 0.25).toFixed(2)));
  });

  zoomOutButton?.addEventListener('click', () => {
    zoomAroundCenter(Math.max(0.3, +(zoomLevel - 0.25).toFixed(2)));
  });

  resetButton?.addEventListener('click', () => {
    zoomLevel = defaultZoomLevel;
    panX = defaultPanX;
    panY = defaultPanY;
    applyTransform();
  });

  stage?.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    if (event.target.closest('.kg-node')) return;
    isDragging = true;
    dragStartX = event.clientX;
    dragStartY = event.clientY;
    dragOriginX = panX;
    dragOriginY = panY;
    stage.classList.add('dragging');
  });

  window.addEventListener('pointermove', event => {
    if (!isDragging) return;
    panX = dragOriginX + (event.clientX - dragStartX);
    panY = dragOriginY + (event.clientY - dragStartY);
    applyTransform();
  });

  window.addEventListener('pointerup', () => {
    if (!isDragging) return;
    isDragging = false;
    stage?.classList.remove('dragging');
  });

  stage?.addEventListener('wheel', event => {
    event.preventDefault();
    const direction = event.deltaY < 0 ? 1 : -1;
    const nextZoom = direction > 0
      ? Math.min(2.8, +(zoomLevel + 0.12).toFixed(2))
      : Math.max(0.3, +(zoomLevel - 0.12).toFixed(2));
    zoomAroundPoint(nextZoom, event.clientX, event.clientY);
  }, { passive: false });

  applyTransform();

  fetch('interactive/extraction_result.json')
    .then(response => {
      if (!response.ok) {
        throw new Error(`Failed to load graph data: ${response.status}`);
      }
      return response.json();
    })
    .then(data => {
      const triples = data.flatMap(item => item.triple_list || []);
      const nodeNames = new Set();
      triples.forEach(triple => {
        nodeNames.add(triple.head);
        nodeNames.add(triple.tail);
      });
      if (statsLabel) {
        statsLabel.textContent = `${nodeNames.size} nodes • ${triples.length} relations`;
      }
      buildGraph(triples);
    })
    .catch(() => {
      detailTitle.textContent = 'Graph unavailable';
      detailType.textContent = 'Load error';
      detailBody.textContent = 'The interactive graph could not be generated from extraction_result.json.';
    });

  window.addEventListener('resize', () => {
    if (!currentLayout) return;
    fitGraphToStage(currentLayout);
  });
}
