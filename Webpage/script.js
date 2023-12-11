// script.js

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