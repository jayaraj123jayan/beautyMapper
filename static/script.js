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

function createStyledList(text) {
    const lines = text.split("\n");
    const listItems = lines.map((line) => {
        const [title, ...content] = line.split(":");
        if (!title || !content) return;
        return `<li><strong>${title
            .trim()
            .replace(/^\d+\.\s*\*\*|\*\*$/g, "")}</strong>: ${content
            .join(":")
            .trim()}</li>`;
    });
    return `<ol>${listItems.join("")}</ol>`;
}

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
                    $("#resultContent").html(
                        createStyledList(data.instructions)
                    );
                    $("#products_result").html("<ul></ul>");
                    data.products.forEach((i) => {
                        $("#products_result ul").append(
                            `<li>${i.name} - <a href="${i.link}" target="_blank">Try this</a></li>`
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
