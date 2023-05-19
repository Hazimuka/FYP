function executePython() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', 'http://localhost:8000/execute', true);
    xhr.onload = function() {
        // if (xhr.status == 200) {
        //     console.log('working');
        // }
    };
    xhr.send();
}

function openImage() {

    var i;

    for(i =1; i<4; i++)
    window.open(`./exported_images/${i}.jpg`, '_blank');
  }