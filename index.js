// GLOBAL DATA HOLDERS
let CARS_DATA = [];
let DATA_MINS = {};
let DATA_MAXS = {};
let PRICE_STD = 1.0;
let MILEAGE_STD = 1.0;
let ENGINE_STD = 10.0;

// UNIQUE CATEGORIES FOR ONE-HOT MAPPINGS
let FUEL_TYPES = [];
let TRANSMISSIONS = [];

// DEFERRED INSTALL PROMPT FOR PWA
let deferredPrompt;

// INITIALIZE APP ON LOAD
document.addEventListener('DOMContentLoaded', () => {
    // 1. Register PWA Service Worker
    registerServiceWorker();

    // 2. Fetch Dataset
    fetch('./data/cars.csv')
        .then(response => {
            if (!response.ok) throw new Error('Dataset file not found.');
            return response.text();
        })
        .then(csvText => {
            CARS_DATA = parseCSV(csvText);
            if (CARS_DATA.length === 0) throw new Error('Parsed dataset is empty.');
            
            // 3. Initialize Statistics and ML Preprocessor Values
            initDataStatistics();

            // 4. Setup Event Listeners
            initEventListeners();

            // 5. Populate Select Dropdowns
            populateCompareDropdowns();

            // 6. Run initial recommendations and analytics
            runMLRecommendations();
            updateMarketAnalytics();
            renderCatalogTable();
        })
        .catch(err => {
            console.error('Initialization error:', err);
            document.getElementById('recommendations-grid').innerHTML = `
                <div class="warning-card">
                    <i class="fa-solid fa-triangle-exclamation"></i>
                    <h4>Failed to Load Dataset</h4>
                    <p>Make sure the application is served via a web server (such as Render or a local server) and that data/cars.csv is present in the repository.<br>${err.message}</p>
                </div>
            `;
        });
});

// ==========================================
// PWA INSTALLATION SERVICES
// ==========================================
function registerServiceWorker() {
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('./sw.js')
            .then(reg => console.log('Service Worker registered successfully!', reg.scope))
            .catch(err => console.error('Service Worker registration failed:', err));
    }

    // Capture install prompt
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        
        // Show install button in header
        const installBtn = document.getElementById('install-btn');
        if (installBtn) {
            installBtn.style.display = 'flex';
        }
    });

    // Custom install button click
    const installBtn = document.getElementById('install-btn');
    if (installBtn) {
        installBtn.addEventListener('click', () => {
            // If iOS Safari
            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
            if (isIOS) {
                document.getElementById('ios-install-modal').style.display = 'flex';
                return;
            }

            if (!deferredPrompt) {
                alert('Installation is fully supported on your browser! Click your browser menu (three dots) and select "Install App" or "Add to Home Screen".');
                return;
            }

            deferredPrompt.prompt();
            deferredPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    console.log('User accepted the PWA install prompt');
                    installBtn.style.display = 'none';
                }
                deferredPrompt = null;
            });
        });
    }

    // Close iOS Help Modal
    const iosClose = document.getElementById('ios-modal-close');
    if (iosClose) {
        iosClose.addEventListener('click', () => {
            document.getElementById('ios-install-modal').style.display = 'none';
        });
    }
}

// ==========================================
// CSV PARSING & DATA PREPROCESSING
// ==========================================
function parseCSV(csvText) {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim() !== '');
    if (lines.length === 0) return [];
    
    // Parse headers, trim spaces/quotes
    const headers = lines[0].split(',').map(h => h.replace(/^["']|["']$/g, '').trim());
    
    return lines.slice(1).map(line => {
        // Safe CSV parser that respects commas inside quotes if any exist (though our dataset is clean)
        let values = [];
        let match;
        let lexer = /"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'|([^,\s]+|[^,]*)/g;
        
        while (match = lexer.exec(line)) {
            let val = match[1] || match[2] || match[3] || '';
            values.push(val.trim());
            if (lexer.lastIndex === line.length && line.endsWith(',')) {
                values.push('');
            }
        }
        
        const obj = {};
        headers.forEach((header, index) => {
            let val = values[index];
            if (val === undefined || val === '') {
                if (header === 'Price' || header === 'Mileage' || header === 'Engine') obj[header] = 0;
                else if (header === 'Seats') obj[header] = 5;
                else obj[header] = 'Unknown';
            } else if (!isNaN(val)) {
                obj[header] = Number(val);
            } else {
                obj[header] = val.replace(/^["']|["']$/g, '');
            }
        });
        return obj;
    });
}

function initDataStatistics() {
    // 1. Gather all numerical boundaries (MinMax Scaler limits)
    const numericCols = ['Price', 'Mileage', 'Seats', 'Engine'];
    
    numericCols.forEach(col => {
        const vals = CARS_DATA.map(car => car[col]);
        DATA_MINS[col] = Math.min(...vals);
        DATA_MAXS[col] = Math.max(...vals);
    });

    // 2. Compute Standard Deviations for Naive Bayes Likelihood Bandwidth
    const prices = CARS_DATA.map(c => c.Price);
    const mileages = CARS_DATA.map(c => c.Mileage);
    const engines = CARS_DATA.map(c => c.Engine);
    
    PRICE_STD = Math.max(standardDeviation(prices), 0.1);
    MILEAGE_STD = Math.max(standardDeviation(mileages), 0.1);
    ENGINE_STD = Math.max(standardDeviation(engines), 10.0);

    // 3. Gather unique categorical types
    FUEL_TYPES = [...new Set(CARS_DATA.map(c => c['Fuel Type']))].sort();
    TRANSMISSIONS = [...new Set(CARS_DATA.map(c => c['Transmission']))].sort();
}

function standardDeviation(arr) {
    const mean = arr.reduce((acc, val) => acc + val, 0) / arr.length;
    const variance = arr.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / arr.length;
    return Math.sqrt(variance);
}

// MinMaxScaler transform equivalent
function scaleValue(val, col) {
    const min = DATA_MINS[col];
    const max = DATA_MAXS[col];
    if (max - min === 0) return 0;
    return (val - min) / (max - min);
}

// ==========================================
// ML ALGORITHMS (KNN & NAIVE BAYES IN JS)
// ==========================================

// K-Nearest Neighbors Engine
function runKNN(query, topN, mode, metric) {
    let dataset = [...CARS_DATA];
    
    if (mode === 'Strict') {
        // 1. Filter dataset strictly by categorical boundaries first
        if (query.fuel !== 'All') {
            dataset = dataset.filter(car => car['Fuel Type'] === query.fuel);
        }
        if (query.transmission !== 'All') {
            dataset = dataset.filter(car => car['Transmission'] === query.transmission);
        }
        if (query.seats !== 'All') {
            dataset = dataset.filter(car => car['Seats'] === query.seats);
        }
        
        if (dataset.length === 0) {
            return []; // No exact matches found
        }
        
        // 2. Scale query numerical coordinates
        const qVec = [
            scaleValue(query.price, 'Price'),
            scaleValue(query.mileage, 'Mileage'),
            scaleValue(query.seats === 'All' ? 5 : query.seats, 'Seats'),
            scaleValue(query.engine, 'Engine')
        ];
        
        // 3. Compute distances purely on numerical features
        const results = dataset.map(car => {
            const carVec = [
                scaleValue(car['Price'], 'Price'),
                scaleValue(car['Mileage'], 'Mileage'),
                scaleValue(car['Seats'], 'Seats'),
                scaleValue(car['Engine'], 'Engine')
            ];
            
            let dist = 0;
            if (metric === 'euclidean') {
                dist = euclideanDistance(qVec, carVec);
            } else {
                dist = cosineDistance(qVec, carVec);
            }
            
            // Calculate Confidence Score
            // Maximum spatial Euclidean distance for 4 dimensions scaled in [0,1] is sqrt(4) = 2.0
            let conf = 0;
            if (metric === 'euclidean') {
                conf = 100 * (1 - (dist / 2.0));
            } else {
                conf = 100 * (1 - dist);
            }
            conf = Math.max(0, Math.min(100, Math.round(conf * 100) / 100));
            
            return { ...car, Distance: dist, ConfidenceScore: conf };
        });
        
        return results.sort((a, b) => b.ConfidenceScore - a.ConfidenceScore).slice(0, topN);
        
    } else { // Soft Mode: Multi-dimensional space including encoded categoricals
        // 1. Construct 10-dimensional Query Vector
        // Columns: Price, Mileage, Seats, Engine, Fuel_Petrol, Fuel_Diesel, Fuel_CNG, Fuel_Electric, Trans_Manual, Trans_Automatic
        const qVec = [
            scaleValue(query.price, 'Price'),
            scaleValue(query.mileage, 'Mileage'),
            scaleValue(query.seats === 'All' ? 5 : query.seats, 'Seats'),
            scaleValue(query.engine, 'Engine')
        ];
        // Add Fuel One-Hot
        FUEL_TYPES.forEach(ft => qVec.push(query.fuel === ft ? 1.0 : 0.0));
        // Add Transmission One-Hot
        TRANSMISSIONS.forEach(t => qVec.push(query.transmission === t ? 1.0 : 0.0));
        
        const numFeatures = qVec.length;
        
        // 2. Compute distances against all vehicles
        const results = dataset.map(car => {
            const carVec = [
                scaleValue(car['Price'], 'Price'),
                scaleValue(car['Mileage'], 'Mileage'),
                scaleValue(car['Seats'], 'Seats'),
                scaleValue(car['Engine'], 'Engine')
            ];
            FUEL_TYPES.forEach(ft => carVec.push(car['Fuel Type'] === ft ? 1.0 : 0.0));
            TRANSMISSIONS.forEach(t => carVec.push(car['Transmission'] === t ? 1.0 : 0.0));
            
            let dist = 0;
            if (metric === 'euclidean') {
                dist = euclideanDistance(qVec, carVec);
            } else {
                dist = cosineDistance(qVec, carVec);
            }
            
            // Calculate Confidence
            let conf = 0;
            if (metric === 'euclidean') {
                const maxDist = Math.sqrt(numFeatures);
                conf = 100 * (1 - (dist / maxDist));
            } else {
                conf = 100 * (1 - dist);
            }
            conf = Math.max(0, Math.min(100, Math.round(conf * 100) / 100));
            
            return { ...car, Distance: dist, ConfidenceScore: conf };
        });
        
        return results.sort((a, b) => b.ConfidenceScore - a.ConfidenceScore).slice(0, topN);
    }
}

// Distance helper functions
function euclideanDistance(v1, v2) {
    let sum = 0;
    for (let i = 0; i < v1.length; i++) {
        sum += Math.pow(v1[i] - v2[i], 2);
    }
    return Math.sqrt(sum);
}

function cosineDistance(v1, v2) {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < v1.length; i++) {
        dotProduct += v1[i] * v2[i];
        normA += Math.pow(v1[i], 2);
        normB += Math.pow(v2[i], 2);
    }
    if (normA === 0 || normB === 0) return 1.0;
    const similarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    return 1.0 - similarity; // Cosine distance = 1 - cosine similarity
}

// Naive Bayes Recommendation Engine
function runNaiveBayes(query, topN, decayFactor) {
    const qPrice = query.price;
    const qMileage = query.mileage;
    const qEngine = query.engine;
    
    const results = CARS_DATA.map(car => {
        let logL = 0.0;
        let logLMax = 0.0;
        
        // --- PRICE LIKELIHOOD (Gaussian continuous) ---
        // Price under or equal is optimal, only penalize over-budget
        if (car['Price'] <= qPrice) {
            // 0.0 log likelihood means probability of 1.0
            logL += 0.0;
            logLMax += 0.0;
        } else {
            const variance = Math.pow(PRICE_STD, 2);
            const pPriceLog = -0.5 * Math.log(2 * Math.PI * variance) - Math.pow(qPrice - car['Price'], 2) / (2 * variance);
            const pPriceMaxLog = -0.5 * Math.log(2 * Math.PI * variance);
            
            logL += pPriceLog;
            logLMax += pPriceMaxLog;
        }
        
        // --- MILEAGE LIKELIHOOD (Gaussian continuous) ---
        // Exceeding target is optimal, only penalize below-target
        if (car['Mileage'] >= qMileage) {
            logL += 0.0;
            logLMax += 0.0;
        } else {
            const variance = Math.pow(MILEAGE_STD, 2);
            const pMileageLog = -0.5 * Math.log(2 * Math.PI * variance) - Math.pow(qMileage - car['Mileage'], 2) / (2 * variance);
            const pMileageMaxLog = -0.5 * Math.log(2 * Math.PI * variance);
            
            logL += pMileageLog;
            logLMax += pMileageMaxLog;
        }
        
        // --- ENGINE CAPACITY LIKELIHOOD (Gaussian continuous) ---
        // Ignore EV motor penalty comparisons
        if (car['Fuel Type'] !== 'Electric' && query.fuel !== 'Electric') {
            const variance = Math.pow(ENGINE_STD, 2);
            const pEngineLog = -0.5 * Math.log(2 * Math.PI * variance) - Math.pow(qEngine - car['Engine'], 2) / (2 * variance);
            const pEngineMaxLog = -0.5 * Math.log(2 * Math.PI * variance);
            
            logL += pEngineLog;
            logLMax += pEngineMaxLog;
        }
        
        // --- CATEGORICAL LIKELIHOOD: FUEL TYPE ---
        if (query.fuel !== 'All') {
            logL += (car['Fuel Type'] === query.fuel) ? Math.log(0.95) : Math.log(0.05);
            logLMax += Math.log(0.95);
        }
        
        // --- CATEGORICAL LIKELIHOOD: TRANSMISSION ---
        if (query.transmission !== 'All') {
            logL += (car['Transmission'] === query.transmission) ? Math.log(0.95) : Math.log(0.05);
            logLMax += Math.log(0.95);
        }
        
        // --- CATEGORICAL LIKELIHOOD: SEATING ---
        if (query.seats !== 'All') {
            logL += (car['Seats'] === query.seats) ? Math.log(0.95) : Math.log(0.05);
            logLMax += Math.log(0.95);
        }
        
        // Log difference to confidence translation
        const logDiff = logLMax - logL;
        const confScore = 100.0 * Math.exp(-logDiff / decayFactor);
        
        return {
            ...car,
            Distance: logDiff,
            ConfidenceScore: Math.max(0, Math.min(100, Math.round(confScore * 100) / 100))
        };
    });
    
    return results.sort((a, b) => b.ConfidenceScore - a.ConfidenceScore).slice(0, topN);
}

// ==========================================
// front-end RENDERING & DYNAMIC INTERFACES
// ==========================================

function runMLRecommendations() {
    const query = getActiveQuery();
    const topN = parseInt(document.getElementById('top-n').value);
    
    // Determine Algorithm
    const isKNN = document.getElementById('algo-knn').classList.contains('active');
    let recommendations = [];
    
    const descText = document.getElementById('algorithm-description');
    
    if (isKNN) {
        const mode = document.getElementById('mode-strict').classList.contains('active') ? 'Strict' : 'Soft';
        const metric = document.getElementById('distance-metric').value;
        const metricDisplay = metric === 'euclidean' ? 'Euclidean Distance' : 'Cosine Similarity';
        
        descText.innerHTML = `The <strong>K-Nearest Neighbors (KNN)</strong> model evaluated <strong>${CARS_DATA.length}</strong> vehicles using <strong>${metricDisplay}</strong> under <strong>${mode} Filtering</strong> and identified the following matches:`;
        
        recommendations = runKNN(query, topN, mode, metric);
    } else {
        const decayFactor = parseFloat(document.getElementById('nb-decay').value);
        
        descText.innerHTML = `The <strong>Naive Bayes Classifier</strong> model calculated continuous <strong>Gaussian Probability Density</strong> (for Price, Mileage, Engine) and Laplace-smoothed categoricals for <strong>${CARS_DATA.length}</strong> vehicles, identifying:`;
        
        recommendations = runNaiveBayes(query, topN, decayFactor);
    }
    
    renderRecommendedCards(recommendations);
    renderDiagnostics(recommendations, isKNN);
    plotEfficiencyComparison(recommendations);
    
    // Compute side-by-side comparison diagnostics
    runAlgorithmComparison();
}

function getActiveQuery() {
    const seatsVal = document.getElementById('seating').value;
    return {
        price: parseFloat(document.getElementById('budget-slider').value),
        fuel: document.getElementById('fuel-type').value,
        transmission: document.getElementById('transmission').value,
        seats: seatsVal === 'All' ? 'All' : parseInt(seatsVal),
        mileage: parseFloat(document.getElementById('mileage-slider').value),
        engine: 1200.0 // target displacement
    };
}

function renderRecommendedCards(recs) {
    const grid = document.getElementById('recommendations-grid');
    grid.innerHTML = '';
    
    if (recs.length === 0) {
        grid.innerHTML = `
            <div class="warning-card">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <h4>No Exact Matches Found</h4>
                <p>No vehicles match your strict categorical criteria (Fuel, Transmission, and Seating combined).<br>
                Please switch the <strong>Recommendation Mode</strong> to <strong>'Soft Matching'</strong> in the sidebar parameters to view closely matched options, or loosen your inputs!</p>
            </div>
        `;
        return;
    }
    
    recs.forEach(car => {
        const ftClass = `badge-${car['Fuel Type'].toLowerCase()}`;
        const transClass = `badge-${car['Transmission'].toLowerCase()}`;
        const engineDisplay = car['Engine'] > 0 ? `${car['Engine']} cc` : 'Electric Drive (EV)';
        const unit = car['Fuel Type'] === 'Electric' ? 'km/charge' : 'kmpl';
        
        const conf = car.ConfidenceScore;
        let confColor = 'conf-high';
        if (conf < 70) confColor = 'conf-low';
        else if (conf < 85) confColor = 'conf-med';
        
        const cardHTML = `
            <div class="car-card">
                <div>
                    <span class="brand-label">${car['Company Name']}</span>
                    <h3 class="car-title">${car['Car Name']}</h3>
                    <div class="badges-row">
                        <span class="badge ${ftClass}">${car['Fuel Type']}</span>
                        <span class="badge ${transClass}">${car['Transmission']}</span>
                    </div>
                    <table class="spec-table">
                        <tr>
                            <td class="spec-label"><i class="fa-solid fa-gas-pump"></i> Efficiency / Range</td>
                            <td class="spec-value">${car['Mileage']} ${unit}</td>
                        </tr>
                        <tr>
                            <td class="spec-label"><i class="fa-solid fa-gear"></i> Engine Displacement</td>
                            <td class="spec-value">${engineDisplay}</td>
                        </tr>
                        <tr>
                            <td class="spec-label"><i class="fa-solid fa-chair"></i> Seating Capacity</td>
                            <td class="spec-value">${car['Seats']} Seater</td>
                        </tr>
                    </table>
                </div>
                <div>
                    <div class="card-bottom">
                        <div class="price-tag">₹${car['Price'].toFixed(2)} L</div>
                        <span class="confidence-indicator ${confColor}"><i class="fa-solid fa-bolt"></i> ${conf}% Match</span>
                    </div>
                    <div class="progress-bar-container">
                        <div class="progress-fill" style="width: ${conf}%"></div>
                    </div>
                </div>
            </div>
        `;
        
        grid.insertAdjacentHTML('beforeend', cardHTML);
    });
}

function renderDiagnostics(recs, isKNN) {
    const body = document.getElementById('diagnostics-body');
    body.innerHTML = '';
    
    const diagText = document.getElementById('diagnostics-text');
    if (isKNN) {
        diagText.innerHTML = `Below are details regarding the mathematical distances calculated by the <strong>KNN</strong> algorithm. <strong>Metric Log</strong> represents absolute spatial coordinate separation (smaller is better).`;
    } else {
        diagText.innerHTML = `Below are details regarding the continuous log-likelihood calculated by the <strong>Naive Bayes</strong> algorithm. <strong>Metric Log</strong> represents log-difference from theoretical perfect fit (smaller is better).`;
    }
    
    if (recs.length === 0) {
        body.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No recommendations computed to analyze.</td></tr>';
        return;
    }
    
    recs.forEach(car => {
        body.insertAdjacentHTML('beforeend', `
            <tr>
                <td><strong>${car['Car Name']}</strong></td>
                <td>₹${car['Price'].toFixed(2)} L</td>
                <td>${car['Mileage']}</td>
                <td style="font-family: monospace;">${car.Distance.toFixed(4)}</td>
                <td><span class="value-badge">${car.ConfidenceScore}%</span></td>
            </tr>
        `);
    });
}

// Plotly Chart: Fuel Efficiency Comparison
function plotEfficiencyComparison(recs) {
    const element = document.getElementById('efficiency-chart');
    if (recs.length === 0) {
        element.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100%; color:var(--text-muted);">No data available</div>';
        return;
    }
    
    const sorted = [...recs].sort((a,b) => a.Mileage - b.Mileage);
    
    const trace = {
        x: sorted.map(c => c.Mileage),
        y: sorted.map(c => c['Car Name']),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: sorted.map(c => c.Mileage),
            colorscale: 'Emerald'
        },
        hovertemplate: '<b>Car:</b> %{y}<br><b>Efficiency:</b> %{x} kmpl/eq<extra></extra>'
    };
    
    const layout = {
        title: {
            text: 'Fuel Efficiency / Range Comparison',
            font: { color: '#ffffff', family: 'Outfit', size: 16 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#9FA6B2' },
            title: { text: 'Mileage (kmpl or equivalent)', font: { color: '#9FA6B2', size: 11 } }
        },
        yaxis: {
            tickfont: { color: '#9FA6B2' }
        },
        margin: { t: 40, b: 40, l: 150, r: 10 },
        height: 320
    };
    
    Plotly.newPlot(element, [trace], layout, { responsive: true, displayModeBar: false });
}

// ==========================================
// TAB 2: COMPARE VEHICLES SPEC
// ==========================================
function populateCompareDropdowns() {
    const selectA = document.getElementById('compare-car-a');
    const selectB = document.getElementById('compare-car-b');
    
    selectA.innerHTML = '';
    selectB.innerHTML = '';
    
    // Sort names alphabetically
    const names = [...new Set(CARS_DATA.map(c => c['Car Name']))].sort();
    
    names.forEach((name, i) => {
        selectA.add(new Option(name, name));
        selectB.add(new Option(name, name));
    });
    
    // Select different default indexes
    selectA.selectedIndex = 0;
    selectB.selectedIndex = Math.min(1, names.length - 1);
    
    // Setup listeners
    selectA.addEventListener('change', renderComparison);
    selectB.addEventListener('change', renderComparison);
    
    renderComparison();
}

function renderComparison() {
    const carAName = document.getElementById('compare-car-a').value;
    const carBName = document.getElementById('compare-car-b').value;
    
    const container = document.getElementById('comparison-result-container');
    container.innerHTML = '';
    
    if (carAName === carBName) {
        container.innerHTML = `
            <div class="warning-card" style="padding:20px;">
                <i class="fa-solid fa-triangle-exclamation" style="font-size:1.8rem;"></i>
                <h4>Identical Vehicles Selected</h4>
                <p>Please select two different models to contrast side-by-side spec diagnostics.</p>
            </div>
        `;
        return;
    }
    
    const carA = CARS_DATA.find(c => c['Car Name'] === carAName);
    const carB = CARS_DATA.find(c => c['Car Name'] === carBName);
    
    if (!carA || !carB) return;
    
    // Helper function to calculate highlight checks
    const compareMetrics = (valA, valB, lowerIsBetter = false) => {
        if (valA === valB) return { a: `<span>${valA}</span>`, b: `<span>${valB}</span>` };
        const aIsBetter = lowerIsBetter ? (valA < valB) : (valA > valB);
        if (aIsBetter) {
            return {
                a: `<span class="better-metric">${valA} ✔</span>`,
                b: `<span class="worse-metric">${valB}</span>`
            };
        } else {
            return {
                a: `<span class="worse-metric">${valA}</span>`,
                b: `<span class="better-metric">${valB} ✔</span>`
            };
        }
    };
    
    const priceRes = compareMetrics(carA.Price, carB.Price, true);
    const mileageRes = compareMetrics(carA.Mileage, carB.Mileage, false);
    const seatsRes = compareMetrics(carA.Seats, carB.Seats, false);
    const engineRes = compareMetrics(carA.Engine, carB.Engine, false);
    
    const engineA = carA.Engine > 0 ? `${carA.Engine} cc` : 'EV Motor';
    const engineB = carB.Engine > 0 ? `${carB.Engine} cc` : 'EV Motor';
    const engineResDisplay = compareMetrics(engineA, engineB, false); // display highlight manually based on logic if needed, but we pass custom scaled check
    
    // Check engine numeric superiority manually to highlight text beautifully
    let finalEngineA = engineA;
    let finalEngineB = engineB;
    if (carA.Engine !== carB.Engine) {
        if (carA.Engine > carB.Engine) {
            finalEngineA = `<span class="better-metric">${engineA} ✔</span>`;
            finalEngineB = `<span class="worse-metric">${engineB}</span>`;
        } else {
            finalEngineA = `<span class="worse-metric">${engineA}</span>`;
            finalEngineB = `<span class="better-metric">${engineB} ✔</span>`;
        }
    }

    const tableHTML = `
        <div class="table-responsive border-container">
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th style="width: 30%;">Feature Specification</th>
                        <th style="width: 35%;">${carA['Car Name']}</th>
                        <th style="width: 35%;">${carB['Car Name']}</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Brand Name</td>
                        <td><strong>${carA['Company Name']}</strong></td>
                        <td><strong>${carB['Company Name']}</strong></td>
                    </tr>
                    <tr>
                        <td>Price (Lakhs)</td>
                        <td style="font-size: 1.1rem;">₹${priceRes.a}</td>
                        <td style="font-size: 1.1rem;">₹${priceRes.b}</td>
                    </tr>
                    <tr>
                        <td>Mileage / Efficiency</td>
                        <td>${mileageRes.a} ${carA['Fuel Type'] === 'Electric' ? 'km/charge' : 'kmpl'}</td>
                        <td>${mileageRes.b} ${carB['Fuel Type'] === 'Electric' ? 'km/charge' : 'kmpl'}</td>
                    </tr>
                    <tr>
                        <td>Fuel Type</td>
                        <td><span class="badge badge-${carA['Fuel Type'].toLowerCase()}">${carA['Fuel Type']}</span></td>
                        <td><span class="badge badge-${carB['Fuel Type'].toLowerCase()}">${carB['Fuel Type']}</span></td>
                    </tr>
                    <tr>
                        <td>Transmission</td>
                        <td><span class="badge badge-${carA['Transmission'].toLowerCase()}">${carA['Transmission']}</span></td>
                        <td><span class="badge badge-${carB['Transmission'].toLowerCase()}">${carB['Transmission']}</span></td>
                    </tr>
                    <tr>
                        <td>Seating Capacity</td>
                        <td>${seatsRes.a} Seater</td>
                        <td>${seatsRes.b} Seater</td>
                    </tr>
                    <tr>
                        <td>Engine Displacement</td>
                        <td>${finalEngineA}</td>
                        <td>${finalEngineB}</td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;
    
    container.innerHTML = tableHTML;
}

// ==========================================
// TAB 3: MARKET ANALYTICS
// ==========================================
function updateMarketAnalytics() {
    // 1. KPI WIDGETS
    document.getElementById('kpi-total-models').innerText = CARS_DATA.length;
    
    const avgPrice = CARS_DATA.reduce((acc, car) => acc + car.Price, 0) / CARS_DATA.length;
    document.getElementById('kpi-avg-price').innerText = `₹${avgPrice.toFixed(2)}L`;
    
    // Best efficiency
    const sortedMil = [...CARS_DATA].sort((a,b) => b.Mileage - a.Mileage)[0];
    document.getElementById('kpi-peak-efficiency').innerText = `${sortedMil.Mileage} ${sortedMil['Fuel Type'] === 'Electric' ? 'km/c' : 'kmpl'}`;
    document.getElementById('kpi-peak-car').innerText = sortedMil['Car Name'];
    
    // Cheapest
    const sortedCheapest = [...CARS_DATA].sort((a,b) => a.Price - b.Price)[0];
    document.getElementById('kpi-cheapest-price').innerText = `₹${sortedCheapest.Price.toFixed(2)}L`;
    document.getElementById('kpi-cheapest-car').innerText = sortedCheapest['Car Name'];

    // 2. PLOT FUEL TYPE DONUT
    plotFuelShareDonut();

    // 3. PLOT BRAND AVERAGE PRICE HORIZONTAL BAR
    plotBrandPriceBar();

    // 4. PLOT BUDGET CARS BAR (filtered dynamically)
    plotBudgetFilterCars();
}

function plotFuelShareDonut() {
    const counts = {};
    CARS_DATA.forEach(c => {
        const ft = c['Fuel Type'];
        counts[ft] = (counts[ft] || 0) + 1;
    });
    
    const brandsList = Object.keys(counts);
    const countVals = Object.values(counts);
    
    // Colors aligning brand colors exactly
    const brandColors = {
        'Petrol': '#4A90E2',
        'Diesel': '#8B572A',
        'CNG': '#F5A623',
        'Electric': '#00E676'
    };
    
    const trace = {
        values: countVals,
        labels: brandsList,
        type: 'pie',
        hole: 0.45,
        marker: {
            colors: brandsList.map(b => brandColors[b] || '#7F8C8D'),
            line: { color: '#070913', width: 2 }
        },
        textposition: 'inside',
        textinfo: 'percent+label',
        hoverinfo: 'label+value'
    };
    
    const layout = {
        title: {
            text: 'Market Share by Fuel Type',
            font: { color: '#ffffff', family: 'Outfit', size: 16 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        legend: {
            orientation: 'h',
            x: 0.5,
            xanchor: 'center',
            y: -0.1,
            yanchor: 'bottom',
            font: { color: '#9FA6B2' }
        },
        margin: { t: 40, b: 60, l: 10, r: 10 },
        height: 380
    };
    
    Plotly.newPlot('fuel-distribution-chart', [trace], layout, { responsive: true, displayModeBar: false });
}

function plotBrandPriceBar() {
    const brandPrices = {};
    const brandCounts = {};
    
    CARS_DATA.forEach(c => {
        const brand = c['Company Name'];
        brandPrices[brand] = (brandPrices[brand] || 0) + c.Price;
        brandCounts[brand] = (brandCounts[brand] || 0) + 1;
    });
    
    const brandAvgs = Object.keys(brandPrices).map(brand => {
        return {
            brand: brand,
            avg: brandPrices[brand] / brandCounts[brand]
        };
    }).sort((a,b) => a.avg - b.avg); // Sort ascending for horizontal bar stack
    
    const trace = {
        x: brandAvgs.map(b => b.avg),
        y: brandAvgs.map(b => b.brand),
        type: 'bar',
        orientation: 'h',
        marker: {
            color: brandAvgs.map(b => b.avg),
            colorscale: 'Viridis'
        },
        hovertemplate: '<b>Brand:</b> %{y}<br><b>Avg Price:</b> ₹%{x:.2f} Lakhs<extra></extra>'
    };
    
    const layout = {
        title: {
            text: 'Average Car Price by Brand (Lakhs)',
            font: { color: '#ffffff', family: 'Outfit', size: 16 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#9FA6B2' },
            title: { text: 'Average Price (in ₹ Lakhs)', font: { color: '#9FA6B2', size: 11 } }
        },
        yaxis: {
            tickfont: { color: '#9FA6B2' }
        },
        margin: { t: 40, b: 40, l: 120, r: 10 },
        height: 380
    };
    
    Plotly.newPlot('price-distribution-chart', [trace], layout, { responsive: true, displayModeBar: false });
}

function plotBudgetFilterCars() {
    const maxBudget = parseFloat(document.getElementById('budget-filter-slider').value);
    
    let filtered = CARS_DATA.filter(c => c.Price <= maxBudget);
    
    // Sort ascending, take top 8 cheapest
    const cheapestEight = filtered.sort((a,b) => a.Price - b.Price).slice(0, 8);
    
    const trace = {
        x: cheapestEight.map(c => c['Car Name']),
        y: cheapestEight.map(c => c.Price),
        type: 'bar',
        marker: {
            color: cheapestEight.map(c => c.Price),
            colorscale: 'Sunsetdark'
        },
        hovertemplate: '<b>Car:</b> %{x}<br><b>Price:</b> ₹%{y:.2f} Lakhs<extra></extra>'
    };
    
    const layout = {
        title: {
            text: `Cheapest Vehicles in Target Budget (under ₹${maxBudget}L)`,
            font: { color: '#ffffff', family: 'Outfit', size: 16 }
        },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickfont: { color: '#9FA6B2', size: 10 },
            tickangle: -20
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#9FA6B2' },
            title: { text: 'Price (₹ Lakhs)', font: { color: '#9FA6B2', size: 11 } }
        },
        margin: { t: 45, b: 65, l: 45, r: 10 },
        height: 360
    };
    
    Plotly.newPlot('budget-cars-chart', [trace], layout, { responsive: true, displayModeBar: false });
}

// ==========================================
// TAB 4: BROWSE CATALOG DATA TABLE
// ==========================================
function renderCatalogTable() {
    const query = document.getElementById('catalog-search').value.toLowerCase();
    const sort = document.getElementById('catalog-sort').value;
    const limit = parseInt(document.getElementById('catalog-limit').value);
    
    let dataset = [...CARS_DATA];
    
    // 1. Search Query filtering
    if (query) {
        dataset = dataset.filter(car => {
            return car['Car Name'].toLowerCase().includes(query) || 
                   car['Company Name'].toLowerCase().includes(query);
        });
    }
    
    // 2. Sort Logic
    if (sort === 'price-low') {
        dataset.sort((a,b) => a.Price - b.Price);
    } else if (sort === 'price-high') {
        dataset.sort((a,b) => b.Price - a.Price);
    } else if (sort === 'mileage-high') {
        dataset.sort((a,b) => b.Mileage - a.Mileage);
    }
    
    // 3. Update counter text
    const displayCount = Math.min(dataset.length, limit);
    document.getElementById('catalog-count-summary').innerHTML = `Showing <strong>${displayCount}</strong> of <strong>${dataset.length}</strong> matching vehicles:`;
    
    // 4. Render Rows
    const tbody = document.getElementById('catalog-table-body');
    tbody.innerHTML = '';
    
    if (dataset.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; color: var(--text-muted);">No matching vehicles found. Adjust your search.</td></tr>';
        return;
    }
    
    const slice = dataset.slice(0, limit);
    slice.forEach(car => {
        const ftClass = `badge-${car['Fuel Type'].toLowerCase()}`;
        const transClass = `badge-${car['Transmission'].toLowerCase()}`;
        const engineDisplay = car.Engine > 0 ? `${car.Engine} cc` : 'Electric';
        
        tbody.insertAdjacentHTML('beforeend', `
            <tr>
                <td><strong>${car['Car Name']}</strong></td>
                <td>${car['Company Name']}</td>
                <td style="color: var(--green-success); font-weight: 600;">₹${car.Price.toFixed(2)}L</td>
                <td>${car.Mileage} ${car['Fuel Type'] === 'Electric' ? 'km/c' : 'kmpl'}</td>
                <td><span class="badge ${ftClass}">${car['Fuel Type']}</span></td>
                <td><span class="badge ${transClass}">${car['Transmission']}</span></td>
                <td>${car.Seats} Seater</td>
                <td>${engineDisplay}</td>
            </tr>
        `);
    });
}

// ==========================================
// GLOBAL EVENT LISTENERS & INTERACTION
// ==========================================
function initEventListeners() {
    // 1. Mobile Sidebar Toggle Drawer
    const sidebar = document.getElementById('app-sidebar');
    const toggleBtn = document.getElementById('sidebar-toggle');
    const closeBtn = document.getElementById('sidebar-close');
    
    if (toggleBtn && sidebar && closeBtn) {
        toggleBtn.addEventListener('click', () => sidebar.classList.add('open'));
        closeBtn.addEventListener('click', () => sidebar.classList.remove('open'));
    }

    // 2. Tab Routing Navigation
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const targetId = tab.getAttribute('data-tab');
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(targetId).classList.add('active');
            
            // Recalculate Plotly sizes inside tabs (very important to maintain layout width)
            window.dispatchEvent(new Event('resize'));
        });
    });

    // 3. Sidebar Slider Values Update & ML trigger
    setupSliderWithCallback('budget-slider', 'budget-value', (v) => `₹${parseFloat(v).toFixed(1)}L`);
    setupSliderWithCallback('mileage-slider', 'mileage-value', (v) => `${parseFloat(v).toFixed(0)} kmpl / eq`);
    setupSliderWithCallback('top-n', 'top-n-value', (v) => v);
    setupSliderWithCallback('nb-decay', 'nb-decay-value', (v) => v);

    // Form inputs change triggers re-calculation
    const inputs = ['budget-slider', 'fuel-type', 'transmission', 'seating', 'mileage-slider', 'distance-metric', 'nb-decay', 'top-n'];
    inputs.forEach(id => {
        document.getElementById(id).addEventListener('input', runMLRecommendations);
    });

    // 4. Algorithm Switch Toggles
    const algoKNN = document.getElementById('algo-knn');
    const algoNB = document.getElementById('algo-nb');
    const knnParams = document.getElementById('knn-params');
    const nbParams = document.getElementById('nb-params');
    
    algoKNN.addEventListener('click', () => {
        algoNB.classList.remove('active');
        algoKNN.classList.add('active');
        nbParams.classList.add('hidden');
        knnParams.classList.remove('hidden');
        runMLRecommendations();
    });
    
    algoNB.addEventListener('click', () => {
        algoKNN.classList.remove('active');
        algoNB.classList.add('active');
        knnParams.classList.add('hidden');
        nbParams.classList.remove('hidden');
        runMLRecommendations();
    });

    // 5. KNN strict/soft toggle clicks
    const modeStrict = document.getElementById('mode-strict');
    const modeSoft = document.getElementById('mode-soft');
    
    modeStrict.addEventListener('click', () => {
        modeSoft.classList.remove('active');
        modeStrict.classList.add('active');
        runMLRecommendations();
    });
    
    modeSoft.addEventListener('click', () => {
        modeStrict.classList.remove('active');
        modeSoft.classList.add('active');
        runMLRecommendations();
    });

    // 6. Analytics budget filter slide
    setupSliderWithCallback('budget-filter-slider', 'budget-filter-value', (v) => `₹${parseFloat(v).toFixed(0)}L`);
    document.getElementById('budget-filter-slider').addEventListener('input', plotBudgetFilterCars);

    // 7. Catalog table filters triggers
    document.getElementById('catalog-search').addEventListener('input', renderCatalogTable);
    document.getElementById('catalog-sort').addEventListener('change', renderCatalogTable);
    document.getElementById('catalog-limit').addEventListener('change', renderCatalogTable);
}

// Slider binding helper
function setupSliderWithCallback(sliderId, badgeId, formatter) {
    const slider = document.getElementById(sliderId);
    const badge = document.getElementById(badgeId);
    if (!slider || !badge) return;
    
    slider.addEventListener('input', (e) => {
        badge.innerText = formatter(e.target.value);
    });
}

// ==========================================
// ALGORITHM COMPARISON SUITE (KNN VS NAIVE BAYES)
// ==========================================

function calculateConstraintAccuracy(car, query) {
    const priceOk = car['Price'] <= query.price;
    const mileageOk = car['Mileage'] >= query.mileage;
    const fuelOk = query.fuel === 'All' || car['Fuel Type'] === query.fuel;
    const transOk = query.transmission === 'All' || car['Transmission'] === query.transmission;
    const seatsOk = query.seats === 'All' || car['Seats'] === query.seats;
    
    const satisfied = (priceOk ? 1 : 0) + (mileageOk ? 1 : 0) + (fuelOk ? 1 : 0) + (transOk ? 1 : 0) + (seatsOk ? 1 : 0);
    return (satisfied / 5) * 100;
}

function runAlgorithmComparison() {
    const query = getActiveQuery();
    const topN = parseInt(document.getElementById('top-n').value);
    
    const mode = document.getElementById('mode-strict').classList.contains('active') ? 'Strict' : 'Soft';
    const metric = document.getElementById('distance-metric').value;
    const decayFactor = parseFloat(document.getElementById('nb-decay').value);
    
    // Run both models concurrently
    const knnRecs = runKNN(query, topN, mode, metric);
    const nbRecs = runNaiveBayes(query, topN, decayFactor);
    
    // 1. Calculate Overlap
    const knnNames = knnRecs.map(c => c['Car Name']);
    const nbNames = nbRecs.map(c => c['Car Name']);
    const overlap = knnNames.filter(name => nbNames.includes(name));
    const overlapPercent = topN > 0 ? (overlap.length / topN) * 100 : 0;
    
    // Update Overlap KPIs
    const overlapVal = document.getElementById('compare-kpi-overlap');
    const overlapDetail = document.getElementById('compare-kpi-overlap-detail');
    if (overlapVal && overlapDetail) {
        overlapVal.innerText = `${overlapPercent.toFixed(0)}%`;
        overlapDetail.innerText = `${overlap.length} out of ${topN} cars common`;
    }
    
    // 2. Calculate average confidence and constraint accuracies
    let knnConfSum = 0;
    let knnAccSum = 0;
    knnRecs.forEach(car => {
        knnConfSum += car.ConfidenceScore;
        knnAccSum += calculateConstraintAccuracy(car, query);
    });
    const knnAvgConf = knnRecs.length > 0 ? (knnConfSum / knnRecs.length) : 0;
    const knnAvgAcc = knnRecs.length > 0 ? (knnAccSum / knnRecs.length) : 0;
    
    let nbConfSum = 0;
    let nbAccSum = 0;
    nbRecs.forEach(car => {
        nbConfSum += car.ConfidenceScore;
        nbAccSum += calculateConstraintAccuracy(car, query);
    });
    const nbAvgConf = nbRecs.length > 0 ? (nbConfSum / nbRecs.length) : 0;
    const nbAvgAcc = nbRecs.length > 0 ? (nbAccSum / nbRecs.length) : 0;
    
    // Update Accuracies KPIs
    const knnAccKpi = document.getElementById('compare-kpi-knn-acc');
    const nbAccKpi = document.getElementById('compare-kpi-nb-acc');
    if (knnAccKpi) knnAccKpi.innerText = `${knnAvgAcc.toFixed(1)}%`;
    if (nbAccKpi) nbAccKpi.innerText = `${nbAvgAcc.toFixed(1)}%`;
    
    // 3. Render side-by-side cards lists
    renderComparisonCards(knnRecs, query, 'knn-compare-list');
    renderComparisonCards(nbRecs, query, 'nb-compare-list');
    
    // 4. Render comparative diagnostics table
    const tbody = document.getElementById('compare-diagnostics-body');
    if (tbody) {
        tbody.innerHTML = '';
        
        if (knnRecs.length === 0 && nbRecs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No recommendations computed to analyze.</td></tr>';
        } else {
            knnRecs.forEach((car, index) => {
                const constAcc = calculateConstraintAccuracy(car, query);
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td><span class="value-badge" style="background:rgba(142,45,226,0.15); color:#AB63FA;">KNN #${index+1}</span></td>
                        <td><strong>${car['Car Name']}</strong></td>
                        <td><span style="font-weight:600; color:var(--green-success);">${car.ConfidenceScore.toFixed(1)}%</span></td>
                        <td style="font-family:monospace;">${car.Distance.toFixed(4)}</td>
                        <td><strong style="color: ${constAcc >= 90 ? 'var(--green-success)' : constAcc >= 70 ? '#FFA726' : 'var(--red-error)'}">${constAcc.toFixed(0)}%</strong></td>
                    </tr>
                `);
            });
            
            nbRecs.forEach((car, index) => {
                const constAcc = calculateConstraintAccuracy(car, query);
                tbody.insertAdjacentHTML('beforeend', `
                    <tr>
                        <td><span class="value-badge" style="background:rgba(0,230,118,0.15); color:#00E676;">NB #${index+1}</span></td>
                        <td><strong>${car['Car Name']}</strong></td>
                        <td><span style="font-weight:600; color:var(--green-success);">${car.ConfidenceScore.toFixed(1)}%</span></td>
                        <td style="font-family:monospace;">${car.Distance.toFixed(4)}</td>
                        <td><strong style="color: ${constAcc >= 90 ? 'var(--green-success)' : constAcc >= 70 ? '#FFA726' : 'var(--red-error)'}">${constAcc.toFixed(0)}%</strong></td>
                    </tr>
                `);
            });
        }
    }
    
    // 5. Plotly Grouped Bar Chart comparison
    plotAlgoComparisonChart(knnAvgConf, knnAvgAcc, nbAvgConf, nbAvgAcc);
}

function renderComparisonCards(recs, query, elementId) {
    const listElement = document.getElementById(elementId);
    if (!listElement) return;
    listElement.innerHTML = '';
    
    if (recs.length === 0) {
        listElement.innerHTML = `
            <div class="warning-card" style="padding:15px; margin: 10px 0; border:1px dashed rgba(255,167,38,0.25);">
                <i class="fa-solid fa-triangle-exclamation" style="font-size:1.5rem; margin-bottom:5px; color:#FFA726;"></i>
                <p style="font-size:0.85rem; color:var(--text-secondary);">No recommendations found under current strict criteria. Switch recommendation mode to Soft Matching or relax preferred filters in sidebar.</p>
            </div>
        `;
        return;
    }
    
    recs.forEach(car => {
        const ftClass = `badge-${car['Fuel Type'].toLowerCase()}`;
        const transClass = `badge-${car['Transmission'].toLowerCase()}`;
        const engineDisplay = car['Engine'] > 0 ? `${car['Engine']} cc` : 'Electric';
        const unit = car['Fuel Type'] === 'Electric' ? 'km/charge' : 'kmpl';
        
        const priceOk = car['Price'] <= query.price;
        const mileageOk = car['Mileage'] >= query.mileage;
        const fuelOk = query.fuel === 'All' || car['Fuel Type'] === query.fuel;
        const transOk = query.transmission === 'All' || car['Transmission'] === query.transmission;
        const seatsOk = query.seats === 'All' || car['Seats'] === query.seats;
        
        const satisfiedCount = (priceOk ? 1 : 0) + (mileageOk ? 1 : 0) + (fuelOk ? 1 : 0) + (transOk ? 1 : 0) + (seatsOk ? 1 : 0);
        const constAcc = (satisfiedCount / 5) * 100;
        
        let accColor = 'conf-high';
        if (constAcc < 70) accColor = 'conf-low';
        else if (constAcc < 90) accColor = 'conf-med';
        
        const conf = car.ConfidenceScore;
        let confColor = 'conf-high';
        if (conf < 70) confColor = 'conf-low';
        else if (conf < 85) confColor = 'conf-med';
        
        const cardHTML = `
            <div class="car-card comparison-car-card" style="margin-bottom: 15px; padding: 18px; min-height: 200px;">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div style="flex: 1; min-width: 0;">
                        <span class="brand-label" style="font-size:0.7rem;">${car['Company Name']}</span>
                        <h4 class="car-title" style="font-size:1.1rem; margin:2px 0 8px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${car['Car Name']}</h4>
                    </div>
                    <div style="text-align:right; flex-shrink: 0; margin-left: 10px;">
                        <span class="confidence-indicator ${confColor}" style="font-size:0.78rem;"><i class="fa-solid fa-bolt"></i> ${conf}% Match</span>
                    </div>
                </div>
                
                <div class="badges-row" style="margin-bottom:8px; gap:4px;">
                    <span class="badge ${ftClass}" style="font-size:0.7rem; padding: 2px 8px;">${car['Fuel Type']}</span>
                    <span class="badge ${transClass}" style="font-size:0.7rem; padding: 2px 8px;">${car['Transmission']}</span>
                </div>
                
                <div class="compare-spec-summary" style="font-size:0.8rem; margin-bottom:10px; border-bottom:1px solid var(--border-glass); padding-bottom:6px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                        <span style="color:var(--text-secondary);">Price:</span>
                        <strong style="color:var(--green-success);">₹${car['Price'].toFixed(2)}L</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:var(--text-secondary);">Efficiency:</span>
                        <strong style="color:var(--text-primary);">${car['Mileage']} ${unit}</strong>
                    </div>
                </div>
                
                <!-- Constraint Checklist -->
                <div class="constraint-checklist" style="font-size:0.72rem; color:var(--text-secondary); background:rgba(255,255,255,0.01); border: 1px solid var(--border-glass); padding:8px; border-radius:8px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span>Constraint Accuracy:</span>
                        <strong class="${accColor}">${constAcc.toFixed(0)}% (${satisfiedCount}/5)</strong>
                    </div>
                    <div class="checklist-items" style="display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; font-size:0.68rem;">
                        <span style="color:${priceOk ? 'var(--green-success)' : 'var(--text-muted)'}; opacity: ${priceOk ? '1' : '0.5'};"><i class="fa-solid ${priceOk ? 'fa-check' : 'fa-xmark'}"></i> Price</span>
                        <span style="color:${mileageOk ? 'var(--green-success)' : 'var(--text-muted)'}; opacity: ${mileageOk ? '1' : '0.5'};"><i class="fa-solid ${mileageOk ? 'fa-check' : 'fa-xmark'}"></i> Mileage</span>
                        <span style="color:${fuelOk ? 'var(--green-success)' : 'var(--text-muted)'}; opacity: ${fuelOk ? '1' : '0.5'};"><i class="fa-solid ${fuelOk ? 'fa-check' : 'fa-xmark'}"></i> Fuel</span>
                        <span style="color:${transOk ? 'var(--green-success)' : 'var(--text-muted)'}; opacity: ${transOk ? '1' : '0.5'};"><i class="fa-solid ${transOk ? 'fa-check' : 'fa-xmark'}"></i> Trans</span>
                        <span style="color:${seatsOk ? 'var(--green-success)' : 'var(--text-muted)'}; opacity: ${seatsOk ? '1' : '0.5'};"><i class="fa-solid ${seatsOk ? 'fa-check' : 'fa-xmark'}"></i> Seats</span>
                    </div>
                </div>
            </div>
        `;
        
        listElement.insertAdjacentHTML('beforeend', cardHTML);
    });
}

function plotAlgoComparisonChart(knnAvgConf, knnAvgAcc, nbAvgConf, nbAvgAcc) {
    const element = document.getElementById('algo-comparison-chart');
    if (!element) return;
    
    const trace1 = {
        x: ['KNN Model', 'Naive Bayes'],
        y: [knnAvgConf, nbAvgConf],
        name: 'Average Match Confidence',
        type: 'bar',
        marker: {
            color: '#AB63FA'
        },
        hovertemplate: '<b>%{x}</b><br>Avg Confidence: %{y:.2f}%<extra></extra>'
    };
    
    const trace2 = {
        x: ['KNN Model', 'Naive Bayes'],
        y: [knnAvgAcc, nbAvgAcc],
        name: 'Avg Constraint Satisfaction',
        type: 'bar',
        marker: {
            color: '#00E676'
        },
        hovertemplate: '<b>%{x}</b><br>Avg Accuracy: %{y:.2f}%<extra></extra>'
    };
    
    const layout = {
        title: {
            text: 'Algorithm Accuracy & Match Score Comparison',
            font: { color: '#ffffff', family: 'Outfit', size: 14 }
        },
        barmode: 'group',
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {
            tickfont: { color: '#9FA6B2' },
        },
        yaxis: {
            gridcolor: 'rgba(255,255,255,0.05)',
            tickfont: { color: '#9FA6B2' },
            title: { text: 'Percentage (%)', font: { color: '#9FA6B2', size: 10 } },
            range: [0, 105]
        },
        legend: {
            font: { color: '#9FA6B2', size: 10 },
            orientation: 'h',
            y: -0.2
        },
        margin: { t: 40, b: 60, l: 45, r: 10 },
        height: 320
    };
    
    Plotly.newPlot(element, [trace1, trace2], layout, { responsive: true, displayModeBar: false });
}

