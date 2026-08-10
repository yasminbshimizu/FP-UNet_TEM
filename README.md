# <p align="center"> Redes Neurais Clássicas e Fuzzy para Segmentação de Imagens de Microscopia Eletrônica de Transmissão: Desempenho e Impactos Socioambientais 🖥️🌎🔥 </p>
<!-- <h2> <center> REDES NEURAIS CLÁSSICAS E FUZZY PARA SEGMENTAÇÃO DE IMAGENS DE MICROSCOPIA ELETRÔNICA DE TRANSMISSÃO: DESEMPENHO E IMPACTOS SOCIOAMBIENTAIS 🖥️🌎🔥</center> </h2> -->
<p align="center"> Trabalho de Conclusão de Curso do Bacharelado em Ciência e Tecnologia da Ilum Escola de Ciência do Centro Nacional de Pesquisa em Energia e Materiais (CNPEM), desenvolvido por Júlia Guedes A. Santos e Yasmin B. Shimizu sob supervisão de Dr. Bruno Focassio e Prof. Dr. Vinícius F. Wasques. </p>

<p align="center">
<img loading="lazy" src="http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge"/>
</p>

<div align="center">
  <img src="https://github.com/user-attachments/assets/ccb6f5f1-0e07-4eb2-aa7c-5f681c57a59c" alt="Descrição da imagem" width="1000"/>
</div>

## 📝 Resumo
A **segmentação de imagens** pode ser definida como o processo de atribuir rótulos a pixels ou regiões com características semelhantes, permitindo a identificação de diferentes estruturas na figura. A fim de obter um bom desempenho nessa tarefa, métodos de **aprendizado profundo** têm sido utilizados, sendo a **U-Net** uma das arquiteturas mais famosas de rede neural para segmentação. 
Um exemplo de aplicação desse modelo são imagens de **microscopia eletrônica de transmissão (TEM)**, que tendem a apresentar bastante rúído e baixo contraste, complexificando a tarefa. Em materiais como o **MoS2** --- com propriedades eletrônicas, ópticas e mecânicas exploradas, por exemplo, na catálise de reações de evolução de hidrogênio ---, a segmentação pode ser utilizada para diferenciar átomos Mo e S. 
Entretanto, objetivando o treinamento de uma rede neural, a obtenção de um grande volume de imagens de microscopia eletrônica rotuladas é dificultada, de modo que uma alternativa é a **simulação** de sistemas dos compostos desejados e suas respectivas imagens de microscopia, a fim de utilizar técnicas de *transfer learning* com o intuito de adaptar tais modelos para o uso em imagens reais. 
É necessário, porém, reconhecer que o uso de inteligência artificial apresenta um elevado **custo ambiental**, com um enormes gasto energético e emissões de CO2 por treinamento. A alternativa para redução de custo computacional da segmentação aqui proposta é a incorporação de **lógica fuzzy** em redes neurais convolucionais através do ***\textit{pooling} fuzzy*** (FP, do inglês *fuzzy pooling*), lidando melhor com ruído nos conjuntos de dados e melhorando o desempenho da rede.
Assim, o presente projeto tem o intuito de comparar **performance e impacto ambiental** entre modelos de aprendizado profundo crisp e fuzzy para segmentação de imagens. Para isso, serão **simuladas imagens de TEM  de MoS2**, as quais serão utilizadas no treino e teste de modelos **U-Net e FP/U-Net** com diferentes profundidades para identificação de átomos. Assim, a hipótese principal do trabalho é que aderir à lógica fuzzy em redes U-Net apresenta desempenho similar ao aumento de profundidade das mesmas, de maneira menos custosa. Os resultados obtidos serão avaliados através de métricas de classificação, como acurácia e IoU, de quantificação, como RMSE, e com **explicabilidade** de modelos, via SHAP e GradCam. Além disso, o tempo, energia e emissões de carbono dos modelos treinados serão estimados utilizando a biblioteca Python CodeCarbon, conectando com debates acerca dos impactos socioambientais causados pela inteligência artificial.

**Palavras-chave:** U-Net; *Pooling* Fuzzy; Microscopia Eletrônica de Transmissão; Explicabilidade; Impactos Socioambientais da Inteligência Artificial; MoS2.

## 🧠 Contribuições dos Colaboradores
**Desenvolvimento:** Júlia Guedes Almeida dos Santos & Yasmin Barbosa Shimizu.

**Colaboradores Ilum:** Letícia Almeida Nunes & Rafael Dalacorte Erdmann.

**Colaboradores LNNano:** Ana Luísa Carvalho Mendonça & Dra. Yasmin Watanabe de Moura.

**Supervisão e Co-Supervisão:** Dr. Bruno Focassio & Prof. Dr. Vinícius Francisco Wasques.

| [<img loading="lazy" src="https://avatars.githubusercontent.com/u/172424779?v=4" width=115><br><sub> Júlia Guedes </sub>](https://github.com/JuliaGuedesASantos)<br> [<sub>Ilum - CNPEM</sub>](https://ilum.cnpem.br/)<br> [<sub>Currículo Lattes</sub>](http://lattes.cnpq.br/9504021537643847)<br> [<sub>Linkedin</sub>](https://www.linkedin.com/in/j%C3%BAlia-guedes-546542283/) | [<img loading="lazy" src="https://avatars.githubusercontent.com/u/171518829?v=4" width=115><br><sub>Yasmin Shimizu</sub>](https://github.com/yasminbshimizu)<br> [<sub>Ilum - CNPEM</sub>](https://ilum.cnpem.br/)<br> [<sub>Currículo Lattes</sub>](https://wwws.cnpq.br/cvlattesweb/PKG_MENU.menu?f_cod=B946BED44B4E2F555F7290AF3E8AF4F3#)<br> [<sub>Linkedin</sub>](https://www.linkedin.com/in/yasminbshimizu/)
| :---: | :---: |
