# ReVuE project website

The interactive companion to **On-Policy Visual Evidence Distillation**, with staged figure animations, token-by-token evidence examples, complete benchmark tables, and the technical report. The site is static and needs no package installation or build step.

Live site: <https://sylvain-wei.github.io/ReVuE/>

## Preview locally

From the repository root:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open <http://127.0.0.1:8000>. Use a local server because the page loads its content and animation assets with `fetch`.

## Publish with GitHub Pages

1. In the repository's **Settings → Pages**, choose **GitHub Actions** as the publishing source.
2. Open **Actions → Project website → Run workflow** on `main`.
3. Use the deployment URL reported by GitHub. Update the README Website badge only after the deployment is available.

Subsequent pushes to `main` that change `website/` or the Pages workflow redeploy the site automatically.

The workflow in [pages.yml](../.github/workflows/pages.yml) uploads only this website directory. See [GitHub's custom workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Files

- `content.json`: Report-derived metadata, abstract, results, findings, and citation.
- `index.html`, `styles.css`, `app.js`: Layout, tables, figure dialogs, and playback controls.
- `assets/branding/`: ReVuE wordmark and blue double-V favicon.
- `assets/fonts/`: Local Space Grotesk and its OFL license.
- `assets/figures/`: Static SVG figures and original PDFs.
- `assets/animations/`: Staged and standalone animations, with timing metadata.
- `assets/documents/revue-paper.pdf`: Technical report. The public copy's Website and Code buttons link to the deployed project website and this repository; the report content and layout are unchanged.

The overview reveals the upper result panels, then **Acquire → Read → Ground → error attribution**. The island example keeps its map and crop visible while revealing 132 tokens and their score changes. Both teacher evaluations score the same fixed student trajectory. Pause, replay, and show-all controls are available; reduced-motion preferences show complete figures.

The Paper button opens [arXiv:2609.36838](https://arxiv.org/abs/2609.36838). The website citation, repository BibTeX, and GitHub citation metadata use the same arXiv record. Data and checkpoint downloads remain TBA.
