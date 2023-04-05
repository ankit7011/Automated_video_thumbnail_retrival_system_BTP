const text = document.querySelector(".second_text");

const textLoad = () => {
    setTimeout(() => {
        text.textContent = "YT Video";
    }, 0);

    setTimeout(() => {
        text.textContent = "Web Series";
    }, 3000);

    setTimeout(() => {
        text.textContent = "Movies";
    }, 6000);
}

textLoad();