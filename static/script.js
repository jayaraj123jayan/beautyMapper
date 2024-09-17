$(document).ready(function () {
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function (e) {
                $("#preview").attr("src", e.target.result).show();
            };

            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#file-upload").change(function () {
        readURL(this);
    });

    $("#uploadForm").on("submit", function (e) {
        e.preventDefault();
        $("#loading").show();
        $("#result-section").hide();
        let formData = new FormData();
        let fileInput = $("#file-upload")[0].files[0];
        formData.append("file", fileInput);
        fetch("/upload", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                $("#loading").hide();
                $("#result-section").hide();
                if (data.instructions && data.products) {
                    $("#resultContent").text(data.instructions);
                    $("#products_result").html("");
                    data.products.forEach((i) => {
                        $("#products_result").append(
                            '<p>Buy this  <a href="' +
                                i.link +
                                '" target="_blank">' +
                                i.name +
                                "</a>.</p>"
                        );
                    });
                } else {
                    $("#resultContent").text("Unexpected response format");
                }
                $("#result").show();
            })
            .catch((error) => {
                console.error("Error:", error);
                $("#resultContent").text("An error occurred: " + error.message);
                $("#result").show();
                $("#loading").hide();
            });
    });
});
function handleFileChange(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const dummy = document.getElementById("dummy-preview");
            const preview = document.getElementById("preview");
            preview.src = e.target.result;
            preview.style.display = "block";
            dummy.style.display = "none";
        };
        reader.readAsDataURL(file);
    }
}
