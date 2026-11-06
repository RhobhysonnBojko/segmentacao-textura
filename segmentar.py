"""
Segmentação por textura com banco de filtros + agrupamento k-means
(implementado à mão, distância euclidiana em 24 dimensões).

Uso:
    python segmentar.py --imagens imagens --k 5 --janela 16 --media 41
"""
import argparse
import glob
import os

import cv2
import numpy as np

CORES = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
                  [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230]],
                 np.uint8)[:, ::-1]  # BGR


# ---------------------------------------------------------------- filtros
def banco_filtros(sigma=1.5, tam=9):
    """8 filtros por escala: 4 barras (0, 45, 90, 135 graus),
    2 bordas (horizontal e vertical) e 2 circulares (spot e LoG)."""
    r = tam // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    gx = -x / sigma ** 2 * g
    gy = -y / sigma ** 2 * g
    gxx = (x ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    gyy = (y ** 2 / sigma ** 4 - 1 / sigma ** 2) * g
    gxy = (x * y / sigma ** 4) * g

    filtros, nomes = [], []
    # barras: 2ª derivada na direção perpendicular à barra
    for ang in (0, 45, 90, 135):
        t = np.deg2rad(ang + 90)
        c, s = np.cos(t), np.sin(t)
        filtros.append(c * c * gxx + 2 * c * s * gxy + s * s * gyy)
        nomes.append(f"barra {ang}°")
    filtros += [gy, gx]  # borda horizontal responde a variação em y
    nomes += ["borda 0°", "borda 90°"]
    # circulares: diferença de gaussianas (spot) e laplaciano da gaussiana
    g2 = np.exp(-(x ** 2 + y ** 2) / (2 * (2 * sigma) ** 2))
    g2 /= g2.sum()
    filtros.append(g - g2)
    nomes.append("spot (DoG)")
    filtros.append(gxx + gyy)
    nomes.append("circular (LoG)")
    # média zero para não responder a regiões lisas
    filtros = [f - f.mean() for f in filtros]
    return filtros, nomes


# ---------------------------------------------------------------- descritor
def descritor(img, filtros, janela, media, escalas=3):
    """Retorna matriz (blocos_y, blocos_x, 24) com a média do módulo das
    respostas em cada janela, nas 3 escalas da pirâmide."""
    h, w = img.shape
    feats = []
    atual = img.astype(np.float32) / 255.0
    for e in range(escalas):
        jan = max(3, (media // (2 ** e)) | 1)   # janela de média proporcional à escala
        for f in filtros:
            resp = np.abs(cv2.filter2D(atual, -1, f))
            m = cv2.blur(resp, (jan, jan))              # média na janela
            feats.append(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR))
        # próxima escala: gaussiano + redução à metade
        atual = cv2.pyrDown(atual)
    F = np.stack(feats, -1)                                # h x w x 24
    by, bx = h // janela, w // janela
    F = F[:by * janela, :bx * janela].reshape(by, janela, bx, janela, -1).mean((1, 3))
    return F


# ---------------------------------------------------------------- k-means
def kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    # inicialização k-means++ (centros espalhados)
    C = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None] - np.array(C)[None]) ** 2).sum(-1), 1)
        C.append(X[rng.choice(len(X), p=d / d.sum())])
    C = np.array(C)
    for _ in range(iters):
        dist = np.sqrt(((X[:, None, :] - C[None]) ** 2).sum(-1))   # euclidiana 24-D
        rot = dist.argmin(1)
        novo = np.array([X[rot == i].mean(0) if np.any(rot == i) else C[i] for i in range(k)])
        if np.allclose(novo, C):
            break
        C = novo
    return rot, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagens", default="imagens")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--janela", type=int, default=16, help="tamanho do bloco rotulado")
    ap.add_argument("--media", type=int, default=41, help="janela da média das respostas")
    ap.add_argument("--saida", default="resultados")
    args = ap.parse_args()
    os.makedirs(args.saida, exist_ok=True)

    arqs = sorted(glob.glob(os.path.join(args.imagens, "*.png")))
    filtros, nomes = banco_filtros()
    print(f"{len(arqs)} imagens | {len(filtros)} filtros x 3 escalas = {3 * len(filtros)} dimensões")

    # salva o banco de filtros para o relatório
    vis = [cv2.resize(cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                      (90, 90), interpolation=cv2.INTER_NEAREST) for f in filtros]
    vis = [cv2.copyMakeBorder(v, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255) for v in vis]
    cv2.imwrite(os.path.join(args.saida, "banco_filtros.png"), cv2.hconcat(vis))

    imgs = [cv2.imread(a, cv2.IMREAD_GRAYSCALE) for a in arqs]
    F = [descritor(im, filtros, args.janela, args.media) for im in imgs]
    forma = F[0].shape[:2]
    X = np.concatenate([f.reshape(-1, 24) for f in F])

    # normalização z-score (cada dimensão com mesmo peso na distância)
    mu, dp = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / dp

    rot, C = kmeans(Xn, args.k)
    rot = rot.reshape(len(imgs), *forma)

    # distância média de cada vetor ao seu centro (coesão)
    print("Blocos por grupo:", np.bincount(rot.ravel(), minlength=args.k))

    for a, im, r in zip(arqs, imgs, rot):
        mapa = CORES[cv2.resize(r.astype(np.uint8), im.shape[::-1], interpolation=cv2.INTER_NEAREST)]
        sobre = cv2.addWeighted(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), 0.55, mapa, 0.45, 0)
        nome = os.path.splitext(os.path.basename(a))[0]
        cv2.imwrite(os.path.join(args.saida, f"{nome}_seg.png"), cv2.hconcat([cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), sobre]))

    np.save(os.path.join(args.saida, "centros.npy"), C * dp + mu)
    print("Centros (z-score) por grupo, primeiras 8 dims (escala 1):")
    np.set_printoptions(precision=2, suppress=True)
    print(C[:, :8])
    print(f"Resultados em {args.saida}/")


if __name__ == "__main__":
    main()
