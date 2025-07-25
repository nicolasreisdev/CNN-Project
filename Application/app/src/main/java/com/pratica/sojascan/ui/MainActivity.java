package com.pratica.sojascan.ui;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;


import com.pratica.sojascan.R;
import com.pratica.sojascan.models.Classifier;

public class MainActivity extends AppCompatActivity {
    private final ImageView imageView = findViewById(R.id.imageView);
    private final Classifier modelCNN = new Classifier();

    // imagem da galeria
    private final ActivityResultLauncher<String> galleryLauncher =
            registerForActivityResult(new ActivityResultContracts.GetContent(), uri -> {
                Log.i("GALERIA", "Selecionando imagem");
                if (uri != null) {
                //    A 'uri' contém o caminho para a imagem selecionada
                    Log.i("GALERIA", "Imagem selecionada");
                    imageView.setImageURI(uri);
                     // chamar  modelo para classificar  a imagem
                    modelCNN.loadModel();
                    modelCNN.preprocess();
                    modelCNN.predict();
                }
            });

     // tirar uma foto (retorna um Bitmap)
    private final ActivityResultLauncher<Void> cameraLauncher =
            registerForActivityResult(new ActivityResultContracts.TakePicturePreview(), bitmap -> {
                Log.i("CAMERA", "Tirando foto");
                if (bitmap != null) {
                //  O 'bitmap' contém a imagem capturada
                    Log.i("CAMERA", "Foto capturada");
                    imageView.setImageBitmap(bitmap);
                    // chamar modelo para classificar a imagem
                    modelCNN.loadModel();
                    modelCNN.preprocess();
                    modelCNN.predict();
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



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }


    public void onSelectCamera(View v) {
        Log.i("CAMERA", "Selecionando camera");
        // Verifica se a permissão de câmera já foi concedida
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            // Se já tiver permissão, abre a câmera diretamente
            Log.i("CAMERA", "Permissão concedida");
            cameraLauncher.launch(null);
        } else {
            Log.i("CAMERA", "Permissão pede permissão");
            // Se não tiver permissão, solicita ao usuário
            requestPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    public void onSelectGalery(View v) {
        Log.i("GALERIA", "Selecionando galeria");
        // imagens da galeria
        galleryLauncher.launch("image/*");
    }

}