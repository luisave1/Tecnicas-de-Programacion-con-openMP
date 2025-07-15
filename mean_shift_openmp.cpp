#include <chrono>               // Para medir tiempo de ejecución
#include <iostream>             // Para imprimir mensajes en consola
#include <opencv2/opencv.hpp>   // Librería OpenCV para manejo de imágenes
#include <cmath>                // Funciones matemáticas (sqrt)
#include <omp.h>                // OpenMP para paralelización en CPU

using namespace cv;
using namespace std;

// ==== Parámetros del algoritmo Mean Shift ====
const float hs = 8.0f;            // Radio espacial (vecindario)
const float hr = 16.0f;           // Radio de color (distancia en Lab)
const int maxIter = 5;            // Máximo número de iteraciones
const float tol_color = 0.3f;     // Tolerancia de cambio en color
const float tol_spatial = 0.3f;   // Tolerancia de cambio espacial

// ==== Estructura para representar un punto 5D (x, y, L, a, b) ====
struct Point5D {
    float x, y, l, a, b;

    Point5D() : x(0), y(0), l(0), a(0), b(0) {}
    Point5D(float x_, float y_, float l_, float a_, float b_)
        : x(x_), y(y_), l(l_), a(a_), b(b_) {}

    // Distancia en espacio de color (Lab)
    float colorDist(const Point5D& p) const {
        return sqrt((l - p.l) * (l - p.l) + (a - p.a) * (a - p.a) + (b - p.b) * (b - p.b));
    }

    // Distancia en espacio espacial (x, y)
    float spatialDist(const Point5D& p) const {
        return sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y));
    }

    // Suma de dos puntos
    Point5D operator+(const Point5D& p) const {
        return Point5D(x + p.x, y + p.y, l + p.l, a + p.a, b + p.b);
    }

    // División escalar (promedio)
    Point5D operator/(float val) const {
        return Point5D(x / val, y / val, l / val, a / val, b / val);
    }
};

// ==== Obtiene un punto 5D desde un píxel en Lab ====
Point5D getPoint5D(int i, int j, const Mat& labImg) {
    Vec3b color = labImg.at<Vec3b>(i, j);
    return Point5D((float)j, (float)i,
        color[0] * 100.0f / 255.0f,   // Escalar L de OpenCV a Lab
        (float)color[1] - 128.0f,     // Ajuste de a
        (float)color[2] - 128.0f);    // Ajuste de b
}

// ==== Algoritmo Mean Shift paralelizado con OpenMP ====
void applyMeanShiftParallel(Mat& labImg) {
    int rows = labImg.rows, cols = labImg.cols;
    Mat result = labImg.clone();  // Crear imagen de salida

    // Paralelizar bucle sobre y,x usando OpenMP
#pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            Point5D current = getPoint5D(y, x, labImg);  // Punto actual
            Point5D prev;
            int iter = 0;

            // Bucle de iteraciones para Mean Shift
            do {
                prev = current;
                Point5D sum(0, 0, 0, 0, 0);  // Acumulador de vecinos válidos
                int count = 0;

                // Recorrer vecindario (hs x hs)
                for (int j = -hs; j <= hs; ++j) {
                    for (int i = -hs; i <= hs; ++i) {
                        int nx = x + i;
                        int ny = y + j;

                        // Verifica si el vecino está dentro de los límites
                        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) {
                            Point5D neighbor = getPoint5D(ny, nx, labImg);
                            // Verifica si vecino está dentro del rango espacial y de color
                            if (current.spatialDist(neighbor) <= hs &&
                                current.colorDist(neighbor) <= hr) {
                                sum = sum + neighbor;
                                count++;
                            }
                        }
                    }
                }

                // Actualiza el punto con el promedio de los vecinos
                if (count > 0) {
                    current = sum / count;
                }

                iter++;
            } while (current.colorDist(prev) > tol_color &&
                     current.spatialDist(prev) > tol_spatial &&
                     iter < maxIter);

            // Convertir de nuevo a formato de OpenCV (Lab)
            int l = static_cast<int>(current.l * 255.0f / 100.0f);
            int a = static_cast<int>(current.a + 128.0f);
            int b = static_cast<int>(current.b + 128.0f);
            result.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(l),
                                           saturate_cast<uchar>(a),
                                           saturate_cast<uchar>(b));
        }
    }

    labImg = result;  // Reemplaza la imagen original por la modificada
}

// ==== Función principal ====
int main() {
    // Cargar imagen de entrada
    Mat Img = imread("C:/Users/LUIS FERNANDO/Pictures/arte/THL.jpg");
    if (Img.empty()) {
        cerr << "Error: No se pudo abrir o encontrar la imagen 'THL.jpg'" << endl;
        return -1;
    }

    // Redimensionar imagen para facilitar el procesamiento
    resize(Img, Img, Size(256, 256), 0, 0, INTER_LINEAR);

    // Mostrar imagen original
    namedWindow("The Original Picture");
    imshow("The Original Picture", Img);

    // Medir tiempo de ejecución
    auto start = chrono::high_resolution_clock::now();

    // Convertir a Lab y aplicar Mean Shift paralelizado
    cvtColor(Img, Img, COLOR_BGR2Lab);
    applyMeanShiftParallel(Img);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "Tiempo de ejecución (OpenMP): " << duration.count() << " ms" << endl;

    // Convertir imagen procesada a BGR para mostrar
    cvtColor(Img, Img, COLOR_Lab2BGR);
    namedWindow("MS Picture (OpenMP)");
    imshow("MS Picture (OpenMP)", Img);

    waitKey(0);
    return 0;
}
