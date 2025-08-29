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

    public static class Result {
        public final String predictedLabel;
        public final float confidence;

        public Result(String predictedLabel, float confidence) {
            this.predictedLabel = predictedLabel;
            this.confidence = confidence;
        }
    }

    public Classifier(Context context, String modelName, String labelName) {
        try {
            // Carrega o modelo e os rótulos do diretório 'assets'.
            Log.i("Teste", "Iniciando modelo");
            model = Module.load(assetFilePath(context, modelName));
            Log.i("Teste", "Model:" + model);
            loadLabels(context, labelName);
            Log.i("Teste", "Labels:" + labels);
            Log.i("Teste", "modelo carregado com sucesso");
        } catch (IOException e) {
            Log.e("Teste", "Erro ao carregar o modelo ou os rótulos.", e);
        }
    }

    private void loadLabels(Context context, String labelName) throws IOException {
        labels = new ArrayList<>();
        InputStream is = context.getAssets().open(labelName);
        Log.i("Teste", "Carregando rótulos");
        try (BufferedReader br = new BufferedReader(new InputStreamReader(is))) {
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
        }
    }

    private float[] softmax(float[] logits) {
        float[] probabilities = new float[logits.length];
        float maxLogit = -Float.MAX_VALUE;
        for (float logit : logits) {
            if (logit > maxLogit) {
                maxLogit = logit;
            }
        }

        float sumExp = 0.0f;
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - maxLogit);
            sumExp += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sumExp;
        }

        return probabilities;
    }

    public Result predict(Bitmap bitmap) {
        if (model == null || labels == null) {
            Log.i("Teste", "Model" + model + "labels" + labels);
            return null;
        }

        // Redimensiona o bitmap para o tamanho de entrada do modelo.
        Log.i("Teste", "Iniciando predição");
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT, true);

        // Converte o Bitmap para um Tensor e normaliza seus valores.
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                resizedBitmap,
                NORM_MEAN,
                NORM_STD
        );

        Log.i("Teste", "Imagem normalizada");

        // Passa o tensor de entrada pelo modelo.
        final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();

        Log.i("Teste", "Imagem transformada em tensor");
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

        Log.i("Teste", "Predição: " + labels.get(maxScoreIdx));

        String predictedLabel = labels.get(maxScoreIdx);

        Log.i("Teste", "Predição feita.");
        //Calcula a confiança usando a função Softmax
        final float[] probabilities = softmax(scores);
        final float confidence = probabilities[maxScoreIdx];

        Log.i("Teste", "Confiança calculada.");
        // Retorna o novo objeto Result
        return new Result(predictedLabel, confidence);
    }

    /**
     * Função utilitária que copia um arquivo dos 'assets' para o armazenamento
     * interno do app e retorna o caminho absoluto desse arquivo. Necessário para
     * que a biblioteca nativa do PyTorch consiga carregar o modelo.
     */
    private static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        Log.i("Teste", "Carregando PyTorch");
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        Log.i("Teste", "Carregando PyTorch 2");
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            catch (IOException e) {
                Log.e("Teste", "Erro ao copiar o arquivo do assets.", e);
            }
            Log.i("Teste", "Modelo carregado");
            return file.getAbsolutePath();
        }
        catch (IOException e) {
            Log.e("Teste", "Erro ao copiar o arquivo do assets.", e);
            throw e;
        }
    }

}
