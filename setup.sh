#unzip api/testes/resultados/testes_automatizados.json.zip -d api/testes/resultados/
#rm api/testes/resultados/testes_automatizados.json.zip

unzip api/conteudo/bancos_vetores/bancos_vetores.zip -d api/conteudo/bancos_vetores
rm api/conteudo/bancos_vetores

cp api/.env.TEMPLATE api/.env
