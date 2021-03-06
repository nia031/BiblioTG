$(document).ready(function(){
    $('#reporte_contenido').click(function(){
        titulos =[];
        scores = [];
        
        $(".isbn").each(function(){
            titulos.push($(this).text());
        });
        $(".score").each(function(){
            scores.push($(this).text());
        });
        scores2 = [];
        $(".score2").each(function(){
            scores2.push($(this).text());
        });
    var barData = {
        labels: titulos,
        datasets: [
            {
                label: "Linear Kernel",
                backgroundColor: 'rgba(220, 220, 220, 0.5)',
                pointBorderColor: "#fff",
                data: scores
            },
            {
                label: "Cosine Similarity",
                backgroundColor: 'rgba(229,57,53,0.5)',
                borderColor: "rgba(229,57,53,0.7)",
                pointBackgroundColor: "rgba(229,57,53,1)",
                pointBorderColor: "#fff",
                data: scores2
            }
        ]
    };

    var barOptions = {
        responsive: true
    };

    var ctx2 = document.getElementById("barChart").getContext("2d");
    new Chart(ctx2, {type: 'bar', data: barData, options:barOptions}); 
    });

    $('#reporte_colaborativo').click(function(){
        if ($('li.svd').hasClass('active')) {            
            titulos =[];
            scores = [];
        
            $(".isbn").each(function(){
                titulos.push($(this).text());
            });
            $(".score").each(function(){
                scores.push($(this).text());
            });
            
            var barData = {
                labels: titulos,
                datasets: [
                    {
                        label: "SVD",
                        backgroundColor: 'rgba(229,57,53, 0.5)',
                        pointBorderColor: "#fff",
                        data: scores
                    }
                ]
            };

            var barOptions = {
                responsive: true
            };

            var ctx = document.getElementById("barChart").getContext("2d");
            new Chart(ctx, {type: 'bar', data: barData, options:barOptions}); 
        }
        
        
        if ($('li.knn').hasClass('active')){
        
            titulos2 =[];
            scores2 = [];
            $(".isbn2").each(function(){
                titulos2.push($(this).text());
            });
            $(".score2").each(function(){
                scores2.push($(this).text());
            });            
            var barData2 = {
                labels: titulos2,
                datasets: [
                    {
                        label: "KNN",
                        backgroundColor: 'rgba(220, 220, 220, 0.5)',
                        pointBorderColor: "#fff",
                        data: scores2
                    }
                ]
            };

            var barOptions2 = {
                responsive: true
            };

            var ctx2 = document.getElementById("barChart").getContext("2d");
            new Chart(ctx2, {type: 'bar', data: barData2, options:barOptions2}); 
        }
    });
});