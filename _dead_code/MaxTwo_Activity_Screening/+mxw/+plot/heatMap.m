function heatMap(map,vals,varargin)
%heatMap plot colormap on array with arbitrary values
%  heatMap(map,vals)
%
%  heatMap(___, 'method',M) specify plotting method. Possible values for
%  M:   - 'gridIP' (Default): Interpolate on grid
%       - 'scatterIP': scatter interpolation
%       - 'scatter': no interpolation
%
%  heatMap(___, 'gridUm',g) Define grid-spacing in um (Default = 5)
%
%  heatMap(___, 'doDots',true) also plot dots at electrode position
%
%  heatMap(___, 'scatterSize',S) also plot dots at electrode position (Default = 100)
%
%  See also neuron


p = inputParser;

p.addParameter('gridUm', 5);
p.addParameter('method', 'scatter');
p.addParameter('doDots', false);
p.addParameter('scatterSize', 100);

p.parse(varargin{:});

args = p.Results;

method = args.method;
gridUm = args.gridUm;



x = map.x;
y = map.y;


switch method
    
    case 'gridIP'
        
        xlin=linspace(min(x),max(x),(max(x)-min(x))/gridUm);
        ylin=linspace(min(y),max(y),(max(y)-min(y))/gridUm);
        [XI,YI] = meshgrid(xlin,ylin);
        
        ZI = griddata(x,y,vals,XI,YI);
        ZI(find(isnan(ZI)))=0;  % set NaN's to zero.
        
        imagesc(xlin,ylin,ZI);
        
    case 'scatterIP'
        
        
        % Option2
        
        F = scatteredInterpolant(x', y', vals', 'linear');
        [xx, yy] = meshgrid(unique(x), unique(y));
        qz = F(xx, yy);
        imagesc([min(x), max(x)], [min(y), max(y)], qz)
        
    case 'scatter'
        
        scatter(x, y, args.scatterSize, vals, 'filled', 's');
        
end

if args.doDots
    
    plot(neur.x,neur.y,'.','color',[0.4 0.4 0.4])
    
end

axis ij
axis equal

