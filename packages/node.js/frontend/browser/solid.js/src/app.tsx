import { Router } from '@solidjs/router';
import { FileRoutes } from '@solidjs/start/router';
import { JSX, Suspense } from 'solid-js';
import './app.css';

const App = (): JSX.Element => {
  return (
    <Router root={(props) => <Suspense>{props.children}</Suspense>}>
      <FileRoutes />
    </Router>
  );
};

export default App;
