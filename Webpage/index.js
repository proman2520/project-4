// script.js

// For index.html
var userInput = document.getElementById('userInput');
var errorMessage = document.getElementById('errorMessage');

userInput.addEventListener('keydown', function (event) {
    if (event.key === 'Enter') {
        event.preventDefault();
        validateInput();
    } else {
        // Clear the error message when the user starts typing
        errorMessage.textContent = '';
    }
});

function validateInput() {
    var options = document.getElementById('optionsList').options;

    // Check if the user input exists in the list of options
    var isValid = Array.from(options).some(function (option) {
        return option.value === userInput.value;
    });

    // Display error message if the input is not valid
    errorMessage.textContent = isValid ? '' : 'Invalid entry. Please choose from the options.';
}

// For movies.html
function filterTable() {
    // Declare variables
    var input, filter, table, tbody, tr, td, i, txtValue;
    input = document.getElementById("searchBar");
    filter = input.value.toUpperCase();
    table = document.getElementById("dataTable");
    tbody = table.getElementsByTagName("tbody")[0];
    tr = tbody.getElementsByTagName("tr");

    // Loop through all table rows and hide those that don't match the search query
    for (i = 0; i < tr.length; i++) {
        tds = tr[i].getElementsByTagName("td");
        for (var j = 0; j < tds.length; j++) {
            td = tds[j];
            if (td) {
                txtValue = td.textContent || td.innerText;
                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                    tr[i].style.display = "";
                    break; // Break the inner loop if a match is found
                } else {
                    tr[i].style.display = "none";
                }
            }
        }
    }
}