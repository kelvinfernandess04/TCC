@echo off
echo ========================================================
echo   GERADOR DE APK OFFLINE (LIBRAS POC) - MODO PRODUCAO
echo ========================================================
echo.
echo [1/3] Ejetando o projeto Expo (Prebuild) para estrutura nativa Android...
call npx expo prebuild --platform android --clean

echo.
echo [2/3] Entrando no diretorio nativo do Android...
cd android

echo.
echo [3/3] Compilando APK Release atraves do Gradle Engine...
call gradlew assembleRelease

echo.
echo ========================================================
echo SUCESSO! 
echo O seu instalador final android (.APK) foi gerado em:
echo POC/android/app/build/outputs/apk/release/app-release.apk
echo ========================================================
pause
