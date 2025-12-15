import json
import http.server
import socketserver
import webbrowser
import os
import csv
from pathlib import Path

# HTML template for the visualization page
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cluster Visualization</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            background-color: #f0f2f5;
            color: #333;
            display: flex;
            height: 100vh;
            overflow: hidden;
        }
        #graph-container {
            flex-grow: 1;
            position: relative;
            background-color: #fff;
            border-right: 1px solid #ddd;
        }
        svg {
            width: 100%;
            height: 100%;
            cursor: grab;
        }
        svg:active {
            cursor: grabbing;
        }
        .node {
            stroke: #fff;
            stroke-width: 1.5px;
            transition: filter 0.15s ease-in-out, stroke-width 0.15s ease-in-out;
            cursor: pointer;
        }
        .node:hover {
            filter: brightness(1.3) drop-shadow(0 0 5px rgba(0, 123, 255, 0.8));
            stroke-width: 3px;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }
        text {
            font-size: 11px;
            font-weight: 500;
            paint-order: stroke;
            stroke: #fff;
            stroke-width: 3px;
            stroke-linecap: butt;
            stroke-linejoin: round;
            pointer-events: none;
            text-anchor: middle;
            dominant-baseline: middle;
        }
        #info-panel {
            width: 350px;
            padding: 20px;
            box-sizing: border-box;
            background-color: #f8f9fa;
            overflow-y: auto;
            transition: width 0.3s ease;
        }
        #info-panel.collapsed {
            width: 0;
            padding: 20px 0;
        }
        #info-panel h2 {
            margin-top: 0;
            font-size: 1.2em;
            color: #007bff;
            border-bottom: 2px solid #007bff;
            padding-bottom: 10px;
        }
        #info-panel p {
            margin: 10px 0;
            line-height: 1.6;
        }
        #info-panel strong {
            color: #555;
        }
        #info-panel .content {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin-top: 15px;
            max-height: 500px;
            overflow-y: auto;
            word-wrap: break-word;
            font-size: 13px;
            line-height: 1.5;
        }

        /* Markdown content styling */
        #info-panel .content h1, 
        #info-panel .content h2, 
        #info-panel .content h3, 
        #info-panel .content h4 {
            margin-top: 1em;
            margin-bottom: 0.5em;
            color: #333;
        }

        #info-panel .content p {
            margin: 0.8em 0;
            line-height: 1.6;
        }

        #info-panel .content ul, 
        #info-panel .content ol {
            margin: 0.8em 0;
            padding-left: 2em;
        }

        #info-panel .content li {
            margin: 0.3em 0;
        }

        #info-panel .content code {
            background-color: #f1f3f4;
            padding: 0.2em 0.4em;
            border-radius: 3px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
            font-size: 0.9em;
        }

        #info-panel .content pre {
            background-color: #f1f3f4;
            padding: 1em;
            border-radius: 5px;
            overflow-x: auto;
            margin: 1em 0;
        }

        #info-panel .content pre code {
            background: none;
            padding: 0;
        }

        /* Math formula styling */
        .MathJax {
            font-size: 1.1em !important;
        }

        mjx-container[jax="CHTML"] {
            line-height: 1.2;
        }
        #toggle-button {
            position: absolute;
            top: 50%;
            right: -1px;
            transform: translateY(-50%);
            width: 20px;
            height: 60px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px 0 0 5px;
            cursor: pointer;
            font-size: 16px;
            z-index: 1001;
            writing-mode: vertical-rl;
            text-orientation: mixed;
            padding: 5px;
            box-sizing: border-box;
        }
        .controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .controls label {
            margin-right: 15px;
            font-size: 14px;
        }
        .controls button {
            margin-left: 10px;
            padding: 5px 10px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }
        .controls button:hover {
            background-color: #0056b3;
        }
    </style>
</head>
<body>
    <div id="graph-container">
        <div class="controls">
            <label for="labels-checkbox">
                <input type="checkbox" id="labels-checkbox" checked> Show Labels
            </label>
            <label for="collision-checkbox">
                <input type="checkbox" id="collision-checkbox" checked> Collision Detection
            </label>
            <button id="reset-positions">Reset Positions</button>
        </div>
        <svg></svg>
    </div>
    <div id="info-panel">
        <button id="toggle-button">Info</button>
        <h2>Node Information</h2>
        <div id="node-info-content">
            <p>Click on a node to see its details here.</p>
        </div>
    </div>

    <script code="https://d3js.org/d3.v7.min.js"></script>
    <script code="https://cdn.jsdelivr.net/npm/markdown-it@13.0.1/dist/markdown-it.min.js"></script>
    <script code="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async code="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
        // Configure MathJax
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true,
                processEnvironments: true,
                packages: {'[+]': ['ams', 'newcommand', 'configmacros']},
                macros: {
                    // Common Chinese math notation
                    'geqslant': '\\geq',
                    'leqslant': '\\leq',
                    'complement': '\\complement',
                    'mathrm': ['\\text{#1}', 1],
                    'dfrac': ['\\frac{#1}{#2}', 2],
                    'tfrac': ['\\frac{#1}{#2}', 2]
                }
            },
            options: {
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }
        };

        // Initialize markdown-it with table support
        const md = window.markdownit({
            html: true,
            linkify: true,
            typographer: true,
            breaks: true
        }).enable(['table']);

        // Function to preprocess LaTeX content for better rendering
        function preprocessLatex(content) {
            try {
                return content
                    // Fix common Chinese math symbols
                    .replace(/（/g, '(')
                    .replace(/）/g, ')')
                    .replace(/，/g, ',')
                    .replace(/。/g, '.')
                    // Handle common LaTeX commands that might not be recognized
                    .replace(/\\mathrm\{([^}]+)\}/g, '\\text{$1}')
                    .replace(/\\geqslant/g, '\\geq')
                    .replace(/\\leqslant/g, '\\leq')
                    // Handle Chinese mathematical symbols
                    .replace(/≥/g, '\\geq')
                    .replace(/≤/g, '\\leq')
                    .replace(/∈/g, '\\in')
                    .replace(/∪/g, '\\cup')
                    .replace(/∩/g, '\\cap')
                    .replace(/π/g, '\\pi')
                    .replace(/α/g, '\\alpha')
                    .replace(/β/g, '\\beta')
                    .replace(/θ/g, '\\theta')
                    .replace(/°/g, '^\\circ')
                    // Clean up any multiple spaces
                    .replace(/\s+/g, ' ')
                    .trim();
            } catch (error) {
                console.warn('LaTeX preprocessing error:', error);
                return content; // Return original content if preprocessing fails
            }
        }
        document.addEventListener('DOMContentLoaded', function() {
            const width = document.getElementById('graph-container').clientWidth;
            const height = document.getElementById('graph-container').clientHeight;
            const svg = d3.select("svg").attr("viewBox", [-width / 2, -height / 2, width, height]);

            const color = d3.scaleOrdinal(d3.schemeCategory10);

            fetch('cluster_data.json')
                .then(response => {
                    console.log('Response status:', response.status);
                    return response.json();
                })
                .then(data => {
                    console.log('Loaded data:', data);
                    console.log('Number of nodes:', data.all_nodes ? data.all_nodes.length : 'No all_nodes property');

                    const { nodes, links } = processData(data);
                    console.log('Processed nodes:', nodes.length, 'links:', links.length);

                    if (nodes.length === 0) {
                        console.error('No nodes to display!');
                        return;
                    }

                    // Calculate node sizes based on content and layer
                    nodes.forEach(node => {
                        const baseSize = 8; // Increased base size
                        const contentFactor = Math.sqrt(node.content.length) / 3; // Reduced divisor for larger nodes
                        const layerFactor = (node.layer + 1) * 3; // Cluster nodes get bigger
                        node.radius = Math.max(baseSize, Math.min(25, baseSize + contentFactor + layerFactor));
                        node.collisionRadius = node.radius + 2; // Add padding for collision
                    });

                    // Set up force simulation with collision detection and layer-based positioning
                    const simulation = d3.forceSimulation(nodes)
                        .force("link", d3.forceLink(links).id(d => d.id)
                            .distance(d => {
                                // Distance based on layer difference and node sizes
                                const sourceLayer = d.source.layer || 0;
                                const targetLayer = d.target.layer || 0;
                                const layerDistance = Math.abs(sourceLayer - targetLayer) * 80 + 50;
                                return layerDistance + d.source.radius + d.target.radius;
                            })
                            .strength(0.8))
                        .force("charge", d3.forceManyBody()
                            .strength(d => {
                                // Stronger repulsion for cluster nodes
                                const baseStrength = -200;
                                return baseStrength * (1 + d.layer * 0.5) * (d.radius / 10);
                            }))
                        .force("collision", d3.forceCollide()
                            .radius(d => d.collisionRadius)
                            .strength(0.9))
                        .force("center", d3.forceCenter(0, 0))
                        .force("radial", d3.forceRadial(d => d.layer * 120, 0, 0).strength(0.1))
                        .on("tick", ticked);

                    const link = svg.append("g")
                        .attr("class", "links")
                        .selectAll("line")
                        .data(links)
                        .join("line")
                        .attr("class", "link");

                    const node = svg.append("g")
                        .attr("class", "nodes")
                        .selectAll("circle")
                        .data(nodes)
                        .join("circle")
                        .attr("class", "node")
                        .attr("r", d => d.radius)
                        .attr("fill", d => color(d.layer))
                        .attr("stroke-width", d => d.layer > 0 ? 2 : 1.5) // Thicker stroke for cluster nodes
                        .on("click", nodeClicked)
                        .call(drag(simulation));

                    const label = svg.append("g")
                        .attr("class", "labels")
                        .selectAll("text")
                        .data(nodes)
                        .join("text")
                        .text(d => {
                            // Show layer info for cluster nodes, just ID for leaf nodes
                            return d.layer > 0 ? `${d.id} (L${d.layer})` : d.id;
                        })
                        .attr("x", 0)
                        .attr("y", 0)
                        .style("font-size", d => `${Math.max(10, d.radius / 2)}px`)
                        .style("font-weight", d => d.layer > 0 ? "bold" : "normal");

                    function ticked() {
                        link
                            .attr("x1", d => d.source.x)
                            .attr("y1", d => d.source.y)
                            .attr("x2", d => d.target.x)
                            .attr("y2", d => d.target.y);

                        node
                            .attr("cx", d => d.x)
                            .attr("cy", d => d.y);

                        label
                            .attr("transform", d => `translate(${d.x},${d.y})`);
                    }

                    // Zoom functionality
                    const zoom = d3.zoom()
                        .scaleExtent([0.1, 8])
                        .on("zoom", (event) => {
                            svg.selectAll('g').attr("transform", event.transform);
                        });
                    svg.call(zoom);

                    // Enhanced drag functionality with hierarchical movement
                    function drag(simulation) {
                        // Helper function to get all descendants of a node
                        function getDescendants(nodeId, allNodes, allLinks) {
                            const descendants = new Set();
                            const queue = [nodeId];

                            while (queue.length > 0) {
                                const currentId = queue.shift();
                                if (descendants.has(currentId)) continue;
                                descendants.add(currentId);

                                // Find children
                                allLinks.forEach(link => {
                                    if (link.source.id === currentId) {
                                        queue.push(link.target.id);
                                    }
                                });
                            }

                            return Array.from(descendants);
                        }

                        function dragstarted(event, d) {
                            if (!event.active) simulation.alphaTarget(0.3).restart();

                            // If this is a cluster node, get all its descendants
                            const draggedNodes = d.layer > 0 ? 
                                getDescendants(d.id, nodes, links).map(id => nodes.find(n => n.id === id)) :
                                [d];

                            // Store initial positions and set as fixed
                            draggedNodes.forEach(node => {
                                if (node) {
                                    node.fx = node.x;
                                    node.fy = node.y;
                                    node._initialDragX = node.x;
                                    node._initialDragY = node.y;
                                }
                            });

                            // Store dragged nodes for use in other drag functions
                            event.subject._draggedNodes = draggedNodes;
                        }

                        function dragged(event, d) {
                            const draggedNodes = event.subject._draggedNodes || [d];
                            const dx = event.x - d._initialDragX;
                            const dy = event.y - d._initialDragY;

                            // Move all dragged nodes by the same offset
                            draggedNodes.forEach(node => {
                                if (node && node._initialDragX !== undefined) {
                                    node.fx = node._initialDragX + dx;
                                    node.fy = node._initialDragY + dy;
                                }
                            });
                        }

                        function dragended(event, d) {
                            if (!event.active) simulation.alphaTarget(0);

                            const draggedNodes = event.subject._draggedNodes || [d];

                            // Release fixed positions
                            draggedNodes.forEach(node => {
                                if (node) {
                                    node.fx = null;
                                    node.fy = null;
                                    delete node._initialDragX;
                                    delete node._initialDragY;
                                }
                            });

                            delete event.subject._draggedNodes;
                        }

                        return d3.drag()
                            .on("start", dragstarted)
                            .on("drag", dragged)
                            .on("end", dragended);
                    }

                    function nodeClicked(event, d) {
                        console.log('Node clicked:', d); // Debug log

                        try {
                            const infoPanel = document.getElementById('node-info-content');

                            // Highlight the clicked node
                            node.style("stroke", n => n.id === d.id ? "#ff4444" : "#fff")
                               .style("stroke-width", n => n.id === d.id ? "3px" : "1.5px");

                            // Get raw content
                            const rawContent = d.content || 'No content available';
                            console.log('Raw content length:', rawContent.length); // Debug log

                            // Process content: preprocess LaTeX first, then render markdown
                            let processedContent;
                            try {
                                const preprocessedContent = preprocessLatex(rawContent);
                                processedContent = md.render(preprocessedContent);
                            } catch (renderError) {
                                console.warn('Markdown/LaTeX processing error:', renderError);
                                // Fallback to plain text
                                processedContent = `<pre>${rawContent}</pre>`;
                            }

                            infoPanel.innerHTML = `
                                <div style="background-color: #e7f3ff; border-left: 4px solid #007bff; padding: 10px; margin-bottom: 15px; border-radius: 4px;">
                                    <h3 style="margin: 0 0 10px 0; color: #0056b3;">Node: ${d.id}</h3>
                                    <p style="margin: 5px 0;"><strong>Layer:</strong> ${d.layer}</p>
                                    <p style="margin: 5px 0;"><strong>Original ID:</strong> ${d.original_id || 'N/A'}</p>
                                    <p style="margin: 5px 0;"><strong>Children:</strong> ${d.children ? d.children.join(', ') : 'None'}</p>
                                </div>
                                <div class="content tex2jax_process" style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 20px; margin-top: 10px;">
                                    <h4 style="margin: 0 0 15px 0; color: #495057; border-bottom: 2px solid #007bff; padding-bottom: 8px;">Content</h4>
                                    <div style="font-size: 14px; line-height: 1.6; color: #212529; max-height: 400px; overflow-y: auto;">
                                        ${processedContent}
                                    </div>
                                    <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #dee2e6; font-size: 12px; color: #6c757d;">
                                        Content length: ${rawContent.length} characters
                                    </div>
                                </div>
                            `;

                            // Ensure the info panel is visible
                            const panel = document.getElementById('info-panel');
                            if (panel.classList.contains('collapsed')) {
                                panel.classList.remove('collapsed');
                            }

                            // Trigger MathJax to re-render the new content
                            if (window.MathJax && window.MathJax.typesetPromise) {
                                window.MathJax.typesetPromise([infoPanel]).catch((err) => {
                                    console.log('MathJax typeset error:', err.message);
                                });
                            }

                        } catch (error) {
                            console.error('Error in nodeClicked:', error);
                        }
                    }

                    // Toggle labels
                    d3.select("#labels-checkbox").on("change", function() {
                        label.style("display", this.checked ? "block" : "none");
                    });

                    // Toggle collision detection
                    d3.select("#collision-checkbox").on("change", function() {
                        if (this.checked) {
                            simulation.force("collision", d3.forceCollide()
                                .radius(d => d.collisionRadius)
                                .strength(0.9));
                        } else {
                            simulation.force("collision", null);
                        }
                        simulation.alpha(0.3).restart();
                    });

                    // Reset positions
                    d3.select("#reset-positions").on("click", function() {
                        nodes.forEach(node => {
                            node.fx = null;
                            node.fy = null;
                        });
                        simulation.alpha(1).restart();
                    });
                })
                .catch(error => {
                    console.error('Error loading or processing data:', error);

                    // Show error message in the graph container
                    svg.append("text")
                        .attr("x", 0)
                        .attr("y", 0)
                        .attr("text-anchor", "middle")
                        .style("font-size", "16px")
                        .style("fill", "#dc3545")
                        .text("Error loading visualization data. Check console for details.");
                });

            function processData(data) {
                console.log('Processing data:', data);

                if (!data || !data.all_nodes) {
                    console.error('Invalid data structure:', data);
                    return { nodes: [], links: [] };
                }

                const allNodes = data.all_nodes;
                const nodes = allNodes.map(d => ({...d})); // Create a copy
                const links = [];
                const nodeMap = new Map(nodes.map(n => [n.id, n]));

                console.log('Node map size:', nodeMap.size);

                nodes.forEach(node => {
                    if (node.children) {
                        node.children.forEach(childId => {
                            if (nodeMap.has(childId)) {
                                links.push({ source: node.id, target: childId });
                            }
                        });
                    }
                });

                console.log('Created', nodes.length, 'nodes and', links.length, 'links');
                return { nodes, links };
            }

            // Info panel toggle
            const toggleButton = document.getElementById('toggle-button');
            const infoPanel = document.getElementById('info-panel');
            toggleButton.addEventListener('click', () => {
                infoPanel.classList.toggle('collapsed');
            });
        });
    </script>
</body>
</html>
"""


def load_clustering_csv(csv_file_path):
    """
    Load clustering hierarchy data from CSV file and convert to JSON format.

    Args:
        csv_file_path (str): Path to the clustering CSV file

    Returns:
        dict: Clustering data in the format expected by the visualization
    """
    all_nodes = []

    try:
        import io

        with open(csv_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # Find the header line (should contain "节点ID,层级,原始ID")
            header_idx = None
            for i, line in enumerate(lines):
                if '节点ID,层级,原始ID' in line:
                    header_idx = i
                    break

            if header_idx is None:
                raise ValueError("Could not find CSV header with expected columns")

            # Process from header line onwards
            remaining_lines = lines[header_idx:]

            # Use csv.DictReader on the remaining content
            csv_content = ''.join(remaining_lines)
            csv_reader = csv.DictReader(io.StringIO(csv_content))

            for row_num, row in enumerate(csv_reader):
                try:
                    # Parse children list - handle None values safely
                    children_str = row.get('子节点', '') or ''
                    children_str = children_str.strip()
                    children = []
                    if children_str:
                        children = [child.strip() for child in children_str.split(',') if child.strip()]

                    # Get content safely
                    content = row.get('内容', '') or ''

                    node = {
                        'id': row['节点ID'],
                        'layer': int(row['层级']),
                        'original_id': row.get('原始ID', '') or '',
                        'content': content,
                        'children': children if children else None
                    }
                    all_nodes.append(node)

                except Exception as row_error:
                    print(f"Warning: Skipping row {row_num} due to error: {row_error}")
                    print(f"Row data: {row}")
                    continue

    except Exception as e:
        print(f"Error loading CSV file {csv_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {'all_nodes': all_nodes}


def create_visualization_server(cluster_data, port=8000):
    """
    Starts a web server to visualize the clustering data.

    Args:
        cluster_data (dict): The clustering data dictionary.
        port (int): The port to run the server on.
    """
    # Create a temporary directory to serve files from
    viz_dir = Path(os.getcwd()) / "viz_temp"
    viz_dir.mkdir(exist_ok=True)

    # Write the cluster data to a JSON file
    data_file_path = viz_dir / "cluster_data.json"
    with open(data_file_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)

    # Write the HTML file
    html_file_path = viz_dir / "index.html"
    with open(html_file_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(viz_dir), **kwargs)

        def do_GET(self):
            if self.path == '/':
                self.path = '/index.html'
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    # Find an available port
    while True:
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
            break
        except OSError:
            print(f"Port {port} is in use, trying next one.")
            port += 1

    url = f"http://localhost:{port}"
    print(f"Serving visualization at: {url}")

    # Open the URL in the default web browser
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\\nShutting down the server.")
        httpd.server_close()
        # Clean up temporary files
        os.remove(data_file_path)
        os.remove(html_file_path)
        os.rmdir(viz_dir)


if __name__ == '__main__':
    # Load real clustering data from CSV files
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output_data"

    # Available clustering files
    available_files = []
    for file_path in output_dir.glob("clustering_hierarchy_*.csv"):
        if file_path.is_file():
            available_files.append(file_path.name)

    # Sort files for consistent ordering
    available_files.sort()

    if not available_files:
        print("No clustering hierarchy CSV files found in output_data directory.")
        print("\nFalling back to mock data for demonstration...")
        # Fallback to simple mock data
        sample_data = {
            "all_nodes": [
                {"id": "root", "layer": 2, "content": "Root Node - This is the top level of the hierarchy",
                 "children": ["cluster_1", "cluster_2"]},
                {"id": "cluster_1", "layer": 1,
                 "content": "First Cluster - Contains mathematical problems about geometry",
                 "children": ["prob_1", "prob_2", "prob_3"]},
                {"id": "cluster_2", "layer": 1,
                 "content": "Second Cluster - Contains mathematical problems about algebra",
                 "children": ["prob_4", "prob_5"]},
                {"id": "prob_1", "layer": 0, "content": "Problem 1: Calculate the area of a circle with radius 5",
                 "children": None},
                {"id": "prob_2", "layer": 0, "content": "Problem 2: Find the volume of a sphere with diameter 10",
                 "children": None},
                {"id": "prob_3", "layer": 0,
                 "content": "Problem 3: Determine the surface area of a cube with side length 3", "children": None},
                {"id": "prob_4", "layer": 0, "content": "Problem 4: Solve the quadratic equation x² + 3x - 4 = 0",
                 "children": None},
                {"id": "prob_5", "layer": 0, "content": "Problem 5: Simplify the expression (2x + 3)(x - 1)",
                 "children": None}
            ]
        }
        create_visualization_server(sample_data)

    elif len(available_files) == 1:
        # Only one file available, use it directly
        filename = available_files[0]
        csv_path = output_dir / filename
        print(f"Loading clustering data from {filename}...")
        cluster_data = load_clustering_csv(csv_path)

        if cluster_data:
            print(f"Successfully loaded {len(cluster_data['all_nodes'])} nodes from {filename}")

            # Print summary information
            layers = {}
            for node in cluster_data['all_nodes']:
                layer = node['layer']
                if layer not in layers:
                    layers[layer] = 0
                layers[layer] += 1

            print("Layer distribution:")
            for layer in sorted(layers.keys()):
                print(f"  Layer {layer}: {layers[layer]} nodes")

            create_visualization_server(cluster_data)
        else:
            print(f"Failed to load data from {filename}")

    else:
        # Multiple files available, let user choose
        print("Multiple clustering hierarchy files found:")
        for i, filename in enumerate(available_files, 1):
            csv_path = output_dir / filename
            print(f"  {i}. {filename}")

        print("\nWhich dataset would you like to visualize?")
        try:
            choice = input(f"Enter number (1-{len(available_files)}) [default: 1]: ").strip()

            if not choice:
                choice_idx = 0
            else:
                choice_idx = int(choice) - 1
                if choice_idx < 0 or choice_idx >= len(available_files):
                    print("Invalid choice. Using first dataset.")
                    choice_idx = 0

            filename = available_files[choice_idx]
            csv_path = output_dir / filename
            print(f"\nLoading clustering data from {filename}...")
            cluster_data = load_clustering_csv(csv_path)

            if cluster_data:
                print(f"Successfully loaded {len(cluster_data['all_nodes'])} nodes from {filename}")

                # Print summary information
                layers = {}
                for node in cluster_data['all_nodes']:
                    layer = node['layer']
                    if layer not in layers:
                        layers[layer] = 0
                    layers[layer] += 1

                print("Layer distribution:")
                for layer in sorted(layers.keys()):
                    print(f"  Layer {layer}: {layers[layer]} nodes")

                create_visualization_server(cluster_data)
            else:
                print(f"Failed to load data from {filename}")

        except (ValueError, KeyboardInterrupt):
            print("\nOperation cancelled or invalid input. Exiting...")
            exit(0)
        except Exception as e:
            print(f"An error occurred: {e}")
            exit(1)
