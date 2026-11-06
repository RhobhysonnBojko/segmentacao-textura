# Segmentação por textura (banco de filtros + k-means)

Atividade 1 – INFO7064 Tópicos em Processamento de Imagens (UFPR).

Segmentação de imagens de objetos sobre uma mesa (livros sobre madeira) usando
apenas métodos clássicos: banco de filtros em 3 escalas, vetores de 24 dimensões
e k-means com distância euclidiana.

## Método
- **Filtros (8 por escala):** barras 0°, 45°, 90°, 135°; bordas 0° e 90°;
  circulares (DoG e LoG). Derivadas da gaussiana, σ = 1,5, 9×9.
- **3 escalas:** gaussiano + redução à metade (`cv2.pyrDown`): 512, 256, 128 px.
- **Descritor:** média do módulo das respostas em janela (41/21/11 px) →
  blocos de 16×16 px com vetor de **24 dimensões** (8 filtros × 3 escalas).
- **Agrupamento:** z-score + k-means (NumPy, k-means++), distância euclidiana 24-D,
  todas as imagens juntas (mesma cor = mesma textura em todas).

## Uso
```bash
pip install -r requirements.txt
python recortar.py      # fotos/ -> imagens/ (36 recortes 512x512 em cinza)
python segmentar.py --k 5 --janela 16 --media 41
```

## Imagens
- `fotos/` – 4 fotos originais de celular
- `imagens/` – 36 recortes 512×512 em cinza (grade 3×3 por foto)
- `resultados/` – `img_XX_seg.png` (original | segmentação) e `banco_filtros.png`

## Grupos (k = 5)
| Cor | Textura |
|---|---|
| vermelho | texto denso |
| amarelo | texto miúdo / transições |
| verde | madeira / textura fraca |
| azul | contornos fortes (lombada, código de barras) |
| laranja | regiões lisas |

## Autor
Rhobhysonn Bojko
