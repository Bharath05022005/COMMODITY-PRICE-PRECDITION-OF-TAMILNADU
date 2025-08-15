// Global variables
let isLoggedIn = false; // Track login state
let predictionChartInstance; // Store chart instance for updates

// Tab navigation functions
function openTab(event, tabId) {
    // Uncomment the following code if you want to restrict access to logged-in users only
    if ((tabId === 'insight' || tabId === 'prediction' || tabId === 'dashboard') && !isLoggedIn) {
         alert("Please log in to access this section.");
         return; // Prevent access if not logged in
     }

    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    
    const tabElement = document.getElementById(tabId);
    if (tabElement) {
        tabElement.classList.add('active');
    }
    
    if (event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
    
    let firstSubTab = document.querySelector(`#${tabId} .sub-tab-button`);
    if (firstSubTab) firstSubTab.click();
    
    if (tabId === 'prediction') {
        const categoryElement = document.getElementById('category');
        if (categoryElement) {
            categoryElement.dispatchEvent(new Event('change'));
        }
    }
}

function openSubTab(event, subTabId) {
    let parentSection = event.currentTarget.closest('.tab-content');
    parentSection.querySelectorAll('.sub-tab-content').forEach(subTab => subTab.classList.remove('active'));
    parentSection.querySelectorAll('.sub-tab-button').forEach(btn => btn.classList.remove('active'));
    document.getElementById(subTabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

function openDashboard() {
    // Uncomment to restrict access to logged-in users
    if (!isLoggedIn) {
         alert("Please log in to access the dashboard.");
         return;
     }
    window.open("https://pandam-vilai.onrender.com/dashboard/", "_blank");  // opens DASHBOARD in new tab
}

// Update username display in account menu
function updateUsernameDisplay(username) {
    const accountUsername = document.getElementById('account-username');
    if (username) {
        accountUsername.textContent = username;
        accountUsername.style.display = 'block';
    } else {
        accountUsername.textContent = '';
        accountUsername.style.display = 'none';
    }
}

// Helper functions for location data
function getDistrictsForState(state) {
    const districtMap = {
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Trichy"],
        "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum"],
        "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kollam"],
        // Add more states and districts as needed
    };
    return districtMap[state] || [];
}

function getMarketsForDistrict(district) {
    const marketMap = {
        "Chennai": ["Koyambedu Wholesale Market", "Chintadripet Fish Market", "Perambur Market"],
        "Coimbatore": ["Ukkadam Market", "Mettupalayam Market", "Pollachi Market"],
        "Bangalore": ["KR Market", "Yeshwanthpur Market", "Electronic City Market"],
        // Add more districts and markets as needed
    };
    return marketMap[district] || [];
}

function getVarietiesForCommodity(commodity) {
    const varietyMap = {
        "Tomato": ["Hybrid", "Country", "Cherry"],
        "Onion": ["Small", "Medium", "Large", "Red"],
        "Potato": ["Russet", "Red", "White", "Sweet"],
        "Arhar (Tur/Red Gram)(Whole)": ["Premium", "Standard", "Regular"],
        "Bengal Gram (Gram)(Whole)": ["Brown", "White", "Green"],
        // Add more commodities and varieties as needed
    };
    return varietyMap[commodity] || ["Standard"];
}

// Initialize event listeners when the DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    // Set the default active tab
    document.getElementById("about").classList.add("active");
    
    // Category selection handler
    document.getElementById('category').addEventListener('change', function() {
        const category = this.value;
        const varietySelect = document.getElementById('variety');
        const varietySection = document.getElementById('variety-section');

        varietySelect.innerHTML = '';

        if (category === 'pulses') {
            const pulses = ["Arhar (Tur/Red Gram)(Whole)","Bengal Gram (Gram)(Whole)","Bengal Gram Dal (Chana Dal)",
            "Black Gram (Urd Beans)(Whole)","Black Gram Dal (Urd Dal)","Green Gram Dal (Moong Dal)","Kabuli Chana (Chickpeas-White)",
            "Kulthi (Horse Gram)","Moath Dal"];
            pulses.forEach(pulse => {
                const option = document.createElement('option');
                option.value = pulse.toLowerCase();
                option.textContent = pulse;
                varietySelect.appendChild(option);
            });
            varietySection.style.display = 'block';
        } else if (category === 'vegetables') {
            const vegetables = ["Ashgourd","Broad Beans","Bitter gourd","Bottle gourd","Brinjal","Cabbage","Capsicum",
            "Carrot","Cluster beans","Coriander (Leaves)","Cauliflower","Drumstick","Green Chilli","Onion","Potato",
            "Pumpkin","Raddish","Snakeguard","Sweet Potato","Tomato"];
            vegetables.forEach(vegetable => {
                const option = document.createElement('option');
                option.value = vegetable.toLowerCase();
                option.textContent = vegetable;
                varietySelect.appendChild(option);
            });
            varietySection.style.display = 'block';
        } else {
            varietySection.style.display = 'none';
        }
    });

    // Get prediction button click handler
    document.getElementById('getPrediction').addEventListener('click', function() {
        const selectedCategory = document.getElementById('category').value;
        const selectedVariety = document.getElementById('variety').value;
        const predictionTable = document.getElementById('predictionTable');
        const predictionChart = document.getElementById('predictionChart').getContext('2d');

        if (!selectedCategory || !selectedVariety) {
            alert("Please select category and variety!");
            return;
        }

        fetch("https://pandam-vilai.onrender.com/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: 'include',
            body: JSON.stringify({ category: selectedCategory, variety: selectedVariety })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                alert("Error from server: " + data.error);
                return;
            }
            // Clear previous table data
            predictionTable.innerHTML = "<tr><th>Date</th><th>Min Price</th><th>Max Price</th><th>Predicted Modal Price</th><th>Price Per kg</th></tr>";
            
            // Prepare data for the chart
            const labels = [];
            const predictedPrices = [];
            
            data.weekly_predictions.forEach(pred => {
                predictionTable.innerHTML += `<tr><td>${pred.Date}</td><td>₹${pred.Min_Price}</td><td>₹${pred.Max_Price}</td><td>₹${pred.Predicted_Modal_Price}</td><td>₹${pred.Price_Per_kg}</td></tr>`;
                labels.push(pred.Date);
                predictedPrices.push(pred.Predicted_Modal_Price);
            });

            // Create or update the chart
            const chartData = {
                labels: labels,
                datasets: [{
                    label: 'Predicted Modal Price',
                    data: predictedPrices,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    borderWidth: 2,
                    fill: true
                }]
            };

            if (predictionChartInstance) {
                predictionChartInstance.data = chartData;
                predictionChartInstance.update();
            } else {
                predictionChartInstance = new Chart(predictionChart, {
                    type: 'line',
                    data: chartData,
                    options: {
                        responsive: true,
                        scales: {
                            x: {
                                ticks: {
                                    font: {
                                        weight: 'bold' // Make x-axis labels bold
                                    }
                                }
                            },
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    font: {
                                        weight: 'bold' // Make y-axis labels bold
                                    }
                                }
                            }
                        }
                    }
                });
            }
        })
        .catch(error => {
            console.error("Error:", error);
            alert("Failed to fetch predictions. Please try again later.");
        });
    });

    // Authentication form elements
    const formTitle = document.getElementById("form-title");
    const authForm = document.getElementById("auth-form");
    const emailField = document.getElementById("email");
    const toggleText = document.getElementById("toggle-form");
    const container = document.querySelector(".container");
    const submitBtn = document.getElementById("submit-btn");

    // Toggle between login and signup forms
    toggleText.addEventListener("click", function () {
        if (emailField.classList.contains("hidden")) {
            // Switch to Sign-up
            formTitle.textContent = "Sign Up";
            emailField.classList.remove("hidden");
            submitBtn.textContent = "Sign Up";
            container.classList.add("signup-mode");
            toggleText.innerHTML = 'Already have an account? <a href="#">Login</a>';
        } else {
            // Switch to Login
            formTitle.textContent = "Login";
            emailField.classList.add("hidden");
            submitBtn.textContent = "Login";
            container.classList.remove("signup-mode");
            toggleText.innerHTML = "Don't have an account? <a href='#'>Sign up</a>";
        }
    });

    // Form submission handler
    authForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const username = document.getElementById("username").value;
        const password = document.getElementById("password").value;
        const email = document.getElementById("email").value;
        const isSignup = !emailField.classList.contains("hidden");

        const endpoint = isSignup ? "signup" : "login";

        fetch(`https://pandam-vilai.onrender.com/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: 'include',
            body: JSON.stringify({
                username,
                password,
                ...(isSignup && { email })
            })
        })
        .then(res => res.json().then(data => ({ ok: res.ok, data })))
        .then(({ ok, data }) => {
            if (!ok) {
                alert(data.error || "Something went wrong");
                return;
            }

            alert(data.message);
            if (!isSignup) {
                isLoggedIn = true;
                updateUsernameDisplay(username);
                openTab({ currentTarget: document.querySelector('.tab-button.active') }, 'insight');
            } else {
                toggleText.click(); // Switch to login mode
            }
        })
        .catch(err => {
            console.error("Error:", err);
            alert("An error occurred");
        });
    });

    // Account menu toggle functionality
    const accountLogo = document.getElementById('account-logo');
    const accountDropdown = document.getElementById('account-dropdown');
    const logoutBtn = document.getElementById('logout-btn');

    accountLogo.addEventListener('click', () => {
        if (accountDropdown.style.display === 'block') {
            accountDropdown.style.display = 'none';
        } else {
            accountDropdown.style.display = 'block';
        }
    });

    // Hide dropdown when clicking outside
    document.addEventListener('click', (event) => {
        if (!accountLogo.contains(event.target) && !accountDropdown.contains(event.target)) {
            accountDropdown.style.display = 'none';
        }
    });

    // Logout functionality
    logoutBtn.addEventListener('click', () => {
        isLoggedIn = false;
        updateUsernameDisplay('');
        accountDropdown.style.display = 'none';
        alert('You have been logged out.');
        openTab({ currentTarget: document.querySelector('.tab-button.active') }, 'login');
    });
    
    // Location data handlers
    // States data
    const states = [
        "Tamil Nadu"
    ];
    
    // Populate states dropdown
    const stateSelect = document.getElementById('state');
    if (stateSelect) {
        states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            stateSelect.appendChild(option);
        });
    }
    
    // Handle state change to load districts
    if (stateSelect) {
        stateSelect.addEventListener('change', function() {
            const selectedState = this.value;
            const districtSelect = document.getElementById('district');
            
            // Clear existing options
            districtSelect.innerHTML = '<option value="">All Districts</option>';
            
            if (selectedState) {
                // Fetch districts based on selected state
                const districts = getDistrictsForState(selectedState);
                districts.forEach(district => {
                    const option = document.createElement('option');
                    option.value = district;
                    option.textContent = district;
                    districtSelect.appendChild(option);
                });
            }
        });
    }
    
    // Handle district change to load markets
    const districtSelect = document.getElementById('district');
    if (districtSelect) {
        districtSelect.addEventListener('change', function() {
            const selectedDistrict = this.value;
            const marketSelect = document.getElementById('market');
            
            // Clear existing options
            marketSelect.innerHTML = '<option value="">All Markets</option>';
            
            if (selectedDistrict) {
                // Fetch markets based on selected district
                const markets = getMarketsForDistrict(selectedDistrict);
                markets.forEach(market => {
                    const option = document.createElement('option');
                    option.value = market;
                    option.textContent = market;
                    marketSelect.appendChild(option);
                });
            }
        });
    }
    
    // Handle commodity change to load varieties
    const commoditySelect = document.getElementById('commodity');
    if (commoditySelect) {
        commoditySelect.addEventListener('change', function() {
            const selectedCommodity = this.value;
            const varietySelect = document.getElementById('variety');
            
            // Clear existing options
            varietySelect.innerHTML = '<option value="">All Varieties</option>';
            
            if (selectedCommodity) {
                // Fetch varieties based on selected commodity
                const varieties = getVarietiesForCommodity(selectedCommodity);
                varieties.forEach(variety => {
                    const option = document.createElement('option');
                    option.value = variety;
                    option.textContent = variety;
                    varietySelect.appendChild(option);
                });
            }
        });
    }
    
    // Form validation
    const downloadForm = document.getElementById('download-form');
    if (downloadForm) {
        downloadForm.addEventListener('submit', function(event) {
            const startDate = document.getElementById('start-date').value;
            const endDate = document.getElementById('end-date').value;
            const specificDate = document.getElementById('specific-date').value;
            
            // Basic validation for date ranges
            if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
                event.preventDefault();
                alert('Start date must be before end date.');
                return false;
            }
            
            // If using both date range and specific date, prioritize date range
            if ((startDate || endDate) && specificDate) {
                event.preventDefault();
                alert('Please use either Date Range OR Specific Date, not both.');
                return false;
            }
            
            return true;
        });
    }
});
