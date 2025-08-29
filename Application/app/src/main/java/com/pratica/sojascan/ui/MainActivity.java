package com.pratica.sojascan.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;


import com.pratica.sojascan.R;
import com.pratica.sojascan.models.Classifier;

import java.io.IOException;
import java.util.Locale;


public class MainActivity extends AppCompatActivity {
    private ImageView imageView;
    private Classifier modelCNN;
    private TextView resultTextView;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        resultTextView = findViewById(R.id.result);
        modelCNN = new Classifier(this, "squeezenet_mobile_2.ptl", "labels.txt");
        Log.i("Teste", "Iniciando app");
    }

    // imagem da galeria
    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                Log.i("Teste", "Selecionando imagem da galeria");
                if (uri != null) {
                    try {
                        // 1. Converte a Uri recebida em um Bitmap
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
                        // 2. Exibe o bitmap no ImageView
                        imageView.setImageBitmap(bitmap);
                        // 3. PASSA O BITMAP PARA O MÉTODO DE CLASSIFICAÇÃO
                        classifyImage(bitmap);
                    } catch (IOException e) {
                        e.printStackTrace();
                        Toast.makeText(this, "Falha ao carregar imagem da galeria.", Toast.LENGTH_SHORT).show();
                    }
                }
            });

    // tirar uma foto (retorna um Bitmap)
    private final ActivityResultLauncher<Void> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicturePreview(), bitmap -> {
                Log.i("Teste", "Tirando foto com a camera");
                if (bitmap != null) {
                    //  O 'bitmap' contém a imagem capturada
                    Log.i("Teste", "Foto capturada");
                    imageView.setImageBitmap(bitmap);
                    // chamar modelo para classificar a imagem
                    classifyImage(bitmap);
                }
            });

    // solicitar a permissão de câmera
    private final ActivityResultLauncher<String> requestPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                // Callback: O que fazer após o usuário responder à solicitação de permissão
                if (isGranted) {
                    //   Permissão concedida, podemos abrir a câmera
                    cameraLauncher.launch(null);
                } else {
                    //   Permissão negada, informe o usuário
                    Toast.makeText(this, "Permissão de câmera negada", Toast.LENGTH_SHORT).show();
                }
            });


    // Método para a classificação
    private void classifyImage(Bitmap bitmap) {
        if (modelCNN == null) {
            Toast.makeText(this, "Erro: Classificador não foi inicializado.", Toast.LENGTH_SHORT).show();
            return;
        }
        String classifying = "Classifying...";
        resultTextView.setText(classifying);
        long inico = System.nanoTime();
        Classifier.Result result = modelCNN.predict(bitmap);
        long fim = System.nanoTime();
        long tempoExecucao = (fim - inico) / 1000000;
        if (result == null) {
            String toastText = "Não foi possível classificar.";
            resultTextView.setText(toastText);
            return;
        }
        String outputText = String.format(Locale.US,
                "Result: %s\nConfiability: %.1f%%\nTime: %d ms",
                result.predictedLabel,
                result.confidence * 100.0f
        );

        resultTextView.setText(outputText);
    }



    public void onSelectCamera(View v) {
        Log.i("Teste", "Selecionando camera");
        // Verifica se a permissão de câmera já foi concedida
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // Se já tiver permissão, abre a câmera diretamente
            Log.i("Teste", "Permissão concedida");
            cameraLauncher.launch(null);
        } else {
            Log.i("Teste", "Permissão pede permissão");
            // Se não tiver permissão, solicita ao usuário
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    public void onSelectGalery(View v) {
        Log.i("Teste", "Selecionando galeria");
        // imagens da galeria
        galleryLauncher.launch("image/*");
    }

}