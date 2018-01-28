/* Display image */
var imageInput = document.getElementById('file-input');
var imageContainer = document.getElementById('image-container');
var maxImageSize = 16000000;

imageInput.addEventListener('change', function() {
    var imageFile = this.files[0];

    if (imageFile.size > maxImageSize) {
        console.error('max size' + maxImageSize + 'exceeded');
    }

    var reader = new FileReader();

    reader.onload = function(e) {
        var image = new Image();
        image.src = e.target.result;

        imageContainer.appendChild(image);
    };

    reader.readAsDataURL(imageFile);
});

/* POST Image */
$('#submit-button').on('click', function() {
    if (!imageInput.files.length) {
        $('#error').text('Please add an image');
        return;
    }

    var fileData = imageInput.files[0];
    var form_data = new FormData();
    form_data.append('file', fileData);

    $.ajax({
        url: '/classify',
        dataType: 'string',
        cache: false,
        contentType: 'file',
        processData: false,
        data: form_data,
        type: 'post'
   })
   .done(function(data) {
        console.log('SUCCESS!');
        console.log('data', data);
   })
   .fail(function( xhr, status, errorThrown ) {
        alert( "Sorry, there was a problem!" );
        console.log( "Error: " + errorThrown );
        console.log( "Status: " + status );
        console.dir( xhr );
    });
});
