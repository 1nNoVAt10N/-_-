declare module 'echarts' {
  export interface EChartsOption {
    [key: string]: any;
  }

  export interface ECharts {
    setOption(option: EChartsOption): void;
    resize(): void;
    dispose(): void;
  }

  export function init(dom: HTMLElement): ECharts;
} 