package com.pratica.sojascan.models;


import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

// Regra de negócio - classe responsável por toda a interação com o modelo PyTorch.
public class Classifier {
    private Module model;
    private List<String> labels;

    private static final float[] NORM_MEAN = new float[]{0.485f, 0.456f, 0.406f};
    private static final float[] NORM_STD = new float[]{0.229f, 0.224f, 0.225f};

    private static final int INPUT_IMAGE_WIDTH = 224;
    private static final int INPUT_IMAGE_HEIGHT = 224;

    public Classifier(Context context, String modelName, String labelName) {
        try {
            // Carrega o modelo e os rótulos do diretório 'assets'.
            model = Module.load(assetFilePath(context, modelName));
            loadLabels(context, labelName);
        } catch (IOException e) {
            Log.e("Classifier", "Erro ao carregar o modelo ou os rótulos.", e);
        }
    }

    private void loadLabels(Context context, String labelName) throws IOException {
        labels = new ArrayList<>();
        InputStream is = context.getAssets().open(labelName);
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
        }
    }

    public String predict(Bitmap bitmap) {
        if (model == null || labels == null) {
            return "Erro: Modelo ou rótulos não carregados.";
        }

        // Redimensiona o bitmap para o tamanho de entrada do modelo.
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, true);

        // Converte o Bitmap para um Tensor e normaliza seus valores.
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                NORM_MEAN,
                NORM_STD
        );

        // Passa o tensor de entrada pelo modelo.
        final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();

        // Pega as pontuações de saída do tensor.
        final float[] scores = outputTensor.getDataAsFloatArray();

        // Encontra o índice da maior pontuação.
        int maxScoreIdx = -1;
        float maxScore = -Float.MAX_VALUE;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxScoreIdx = i;
            }
        }

        // Retorna o nome da classe correspondente ao índice encontrado.
        return labels.get(maxScoreIdx);
    }

    /**
     * Função utilitária que copia um arquivo dos 'assets' para o armazenamento
     * interno do app e retorna o caminho absoluto desse arquivo. Necessário para
     * que a biblioteca nativa do PyTorch consiga carregar o modelo.
     */
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
